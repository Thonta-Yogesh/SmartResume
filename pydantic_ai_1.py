from __future__ import annotations
import re
import logging
from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import os
import json
from typing import List
from supabase import Client

# Import your agent framework components.
from pydantic_ai import Agent, RunContext
from sentence_transformers import SentenceTransformer

# NLP Preprocessing (Paper Section III-B)
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Named Entity Recognition (Paper Section IV-B)
import spacy

load_dotenv()

# ─── Logging Setup ────────────────────────────────────────────
logger = logging.getLogger("SmartResume.PydanticAI")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(_handler)

# ─── NLTK / spaCy bootstrap ──────────────────────────────────
for _res in ["punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{_res}" if "punkt" in _res
                       else f"corpora/{_res}")
    except LookupError:
        nltk.download(_res, quiet=True)

try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download as _spacy_dl
    _spacy_dl("en_core_web_sm")
    _nlp = spacy.load("en_core_web_sm")

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

# Set up the model name for generation.
llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    embedding_model: SentenceTransformer


# System prompt updated for a resume search and analysis expert.
system_prompt = """
You are an expert in resume search and analysis. You have access to a resume database containing structured resume chunks.
Each chunk includes:
  - URL (the unique resume identifier)
  - Chunk number
  - Title
  - Summary
  - Content (the actual text of the chunk)
  - Metadata and embeddings

Your job is to help the user find the most relevant resume information based on their query.
When presenting results, provide a clear analysis of each candidate including:
1. How well they match the query requirements
2. Key qualifications and skills
3. Relevant experience
4. An overall fit assessment

If you cannot find a match, be honest and let the user know.
"""

pydantic_ai_expert = Agent(
    f'openai:{llm}',
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2,
    defer_model_check=True
)


# Updated embedding function that uses SentenceTransformers.
async def get_embedding(text: str, embedding_model: SentenceTransformer) -> List[float]:
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(None, embedding_model.encode, text)
    return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)


def preprocess_text(text: str) -> str:
    """
    NLP preprocessing pipeline (Paper Section III-B):
      Tokenization → stopword removal → lemmatization.
    """
    tokens = word_tokenize(text.lower())
    filtered = [tok for tok in tokens if tok.isalpha() and tok not in _stop_words]
    lemmatized = [_lemmatizer.lemmatize(tok) for tok in filtered]
    return " ".join(lemmatized)


def extract_skills_ner(text: str) -> List[str]:
    """
    Extract skills using spaCy NER (Paper Section IV-B).
    """
    doc = _nlp(text)
    ner_skills = set()
    tech_labels = {"ORG", "PRODUCT", "LANGUAGE", "WORK_OF_ART"}
    for ent in doc.ents:
        if ent.label_ in tech_labels and len(ent.text) > 1:
            ner_skills.add(ent.text.strip())
    return sorted(ner_skills)


def extract_experience_years(content: str) -> float:
    """
    Extract years of experience from resume content using regex and text analysis.
    Looks for patterns like '5 years', '3+ years of experience', '2019-2024', etc.
    """
    total_years = 0.0

    # Pattern 1: "X years of experience" or "X+ years"
    year_patterns = re.findall(
        r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of\s+experience)?',
        content,
        re.IGNORECASE,
    )
    if year_patterns:
        total_years = max(float(y) for y in year_patterns)

    # Pattern 2: Date ranges like "2019 - 2024" or "2019 – Present"
    date_ranges = re.findall(
        r'(20\d{2}|19\d{2})\s*[-–—]\s*(20\d{2}|19\d{2}|[Pp]resent|[Cc]urrent)',
        content,
    )
    for start_str, end_str in date_ranges:
        start = int(start_str)
        if end_str.lower() in ('present', 'current'):
            from datetime import datetime
            end = datetime.now().year
        else:
            end = int(end_str)
        span = end - start
        if 0 < span <= 40:
            total_years = max(total_years, float(span))

    return total_years


def calculate_tech_match_score(content: str, required_technologies: List[str]) -> float:
    """
    Calculate a match score based on required technologies.
    Returns a score from 0.0 to 1.0 representing the fraction of required
    technologies found in the resume content.
    """
    if not required_technologies:
        return 0.0

    content_lower = content.lower()
    matches = sum(1 for tech in required_technologies if tech.lower() in content_lower)
    return round(matches / len(required_technologies), 2)


@pydantic_ai_expert.tool
async def retrieve_relevant_resumes(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant resume chunks based on the user's query using retrieval-augmented generation (RAG).
    """
    try:
        # Preprocess query before embedding (Paper §III-B)
        preprocessed_query = preprocess_text(user_query)
        query_embedding = await get_embedding(preprocessed_query, ctx.deps.embedding_model)
        logger.info("Query preprocessed: '%s' → '%s'",
                    user_query[:80], preprocessed_query[:80])

        # Extract technology keywords from the query for scoring
        common_techs = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue',
            'node', 'sql', 'aws', 'docker', 'kubernetes', 'machine learning',
            'deep learning', 'nlp', 'tensorflow', 'pytorch', 'c++', 'c#',
            'go', 'rust', 'typescript', 'html', 'css', 'mongodb', 'postgresql',
            'django', 'flask', 'spring', 'git', 'linux', 'azure', 'gcp',
        ]
        query_lower = user_query.lower()
        query_technologies = [t for t in common_techs if t in query_lower]

        result = ctx.deps.supabase.rpc(
            'match_resumes',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {}
            }
        ).execute()

        if not result.data:
            return "No relevant resume information found for your query."

        # Process and score results
        processed_results = []
        for doc in result.data:
            experience_years = extract_experience_years(doc.get('content', ''))
            tech_match_score = calculate_tech_match_score(
                doc.get('content', ''),
                query_technologies if query_technologies else ['python', 'java', 'react']
            )

            # NER-based skill extraction (Paper §IV-B)
            ner_skills = extract_skills_ner(doc.get('content', ''))

            processed_results.append({
                'doc': doc,
                'experience_years': experience_years,
                'tech_match_score': tech_match_score,
                'ner_skills': ner_skills
            })

        # Sort by experience years (descending) and then by technology match score
        processed_results.sort(
            key=lambda x: (x['experience_years'], x['tech_match_score']),
            reverse=True
        )

        # Format output
        formatted_chunks = []
        for idx, proc_result in enumerate(processed_results):
            doc = proc_result['doc']
            summary = doc.get('summary', '')
            exp_display = f"{proc_result['experience_years']:.0f}" if proc_result['experience_years'] > 0 else "N/A"
            chunk_text = f"""
📄 Resume: {doc.get('url', 'Unknown')} (Chunk {doc.get('chunk_number', 0)})
👤 Title: {doc.get('title', 'Unknown')}
📅 Experience: {exp_display} years
🎯 Tech Match Score: {proc_result['tech_match_score']:.0%}
🧠 NER Skills: {', '.join(proc_result['ner_skills']) if proc_result['ner_skills'] else 'N/A'}
📝 Summary: {summary}

{doc.get('content', '')}
"""
            formatted_chunks.append(chunk_text.strip())

        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        logger.error("Error retrieving resume chunks: %s", e)
        return f"Error retrieving resume chunks: {str(e)}"


@pydantic_ai_expert.tool
async def list_uploaded_resumes(ctx: RunContext[PydanticAIDeps]) -> str:
    """
    List all unique resume URLs (acting as identifiers) from the database.
    """
    try:
        result = ctx.deps.supabase.from_('resumes')\
            .select('url')\
            .execute()

        if not result.data:
            return "No resumes found in the database."

        urls = sorted(set(doc['url'] for doc in result.data))
        formatted = "\n".join(f"  📄 {url}" for url in urls)
        return f"Found {len(urls)} resumes in the database:\n{formatted}"

    except Exception as e:
        logger.error("Error retrieving resume URLs: %s", e)
        return f"Error retrieving resume list: {str(e)}"


@pydantic_ai_expert.tool
async def get_resume_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a resume by combining all its chunks, ordered by chunk_number.
    """
    try:
        result = ctx.deps.supabase.from_('resumes')\
            .select('title, content, chunk_number')\
            .eq('url', url)\
            .order('chunk_number')\
            .execute()

        if not result.data:
            return f"No content found for resume with URL: {url}"

        page_title = result.data[0]['title']
        formatted_content = [f"# Resume: {url} - {page_title}\n"]

        for chunk in result.data:
            formatted_content.append(chunk['content'])

        return "\n\n".join(formatted_content)

    except Exception as e:
        logger.error("Error retrieving resume content: %s", e)
        return f"Error retrieving resume content: {str(e)}"
