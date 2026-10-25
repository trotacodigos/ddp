"""Prompt for APE_exp document annotation (Fig. 8)."""

SYSTEM_PROMPT = (
    "You are a discourse analyst. Your task is to analyze a document and extract "
    "its key discourse attributes. Be concise and precise. Output ONLY valid JSON."
)

GENRE_EXAMPLES = {
    "news": {
        "text_type": "news report, investigative article, opinion piece, editorial, "
                     "press release, interview, feature article",
        "domain": "politics, diplomacy, economics, finance, science, technology, environment, "
                  "health, sports, culture, society, crime, international affairs, military, education",
    },
    "social": {
        "text_type": "social media post, blog post, online comment, forum thread, "
                     "product review, newsletter, personal essay",
        "domain": "lifestyle, travel, food, fashion, entertainment, gaming, parenting, "
                  "health and wellness, personal finance, technology, politics, sports",
    },
    "literary": {
        "text_type": "literary narrative, literary dialogue, short story, novel excerpt, "
                     "poetry, drama, personal memoir, historical fiction",
        "domain": "coming-of-age, romance, historical, thriller, family, war, "
                  "social commentary, philosophical, fantasy, biographical",
    },
}

OUTPUT_SCHEMA = """{
  "genre_and_domain": {
    "text_type": "<select from examples above>",
    "domain": "<select from examples above>"
  },
  "participant_relationships": {
    "identified": <true | false>,
    "participants": [
      {
        "role": "<e.g. narrator, interviewer, character, customer, agent>",
        "description": "<brief description if identifiable>"
      }
    ],
    "relationship_type": "<e.g. interviewer--interviewee, narrator--character, customer--agent, none identifiable>"
  },
  "register_and_formality": {
    "overall_register": "<one of: formal | semi-formal | informal | mixed>",
    "notes": "<any additional register observations relevant to translation>"
  }
}"""

RULES = """Rules:
- Use the provided genre label as the primary guide for text_type and domain selection.
- You may use values outside the examples if none fits, but prefer the listed options.
- If participant relationships cannot be identified, set "identified" to false and leave "participants" as an empty list.
- Do not infer information not present in the document.
- Output ONLY the JSON object, no preamble or explanation."""


def build_prompt(genre: str, src_doc: str) -> tuple[str, str]:
    ex = GENRE_EXAMPLES.get(genre, {"text_type": "other", "domain": "other"})
    user = (
        f'The following source document belongs to the "{genre}" genre. Analyze its content '
        f"and extract the three discourse attributes listed below. Base your analysis solely "
        f"on the document content.\n\n"
        f"[Source Document]\n{src_doc}\n\n"
        f"Use the genre label to select the most appropriate text_type and domain from the "
        f"examples provided, then extract all attributes and output a JSON object.\n\n"
        f"Genre: {genre}\n"
        f"  text_type examples: {ex['text_type']}\n"
        f"  domain examples: {ex['domain']}\n\n"
        f"Output the following JSON structure:\n\n{OUTPUT_SCHEMA}\n\n{RULES}"
    )
    return SYSTEM_PROMPT, user
