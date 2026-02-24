
# Common formatting rules to avoid repetition
HTML_FORMAT_RULES = "Use HTML tags ONLY: <h3>, <p>, <i>, <strong>, <ul>, <li>. NO markdown."

clean_transcript_prompt = """Transform this Whisper transcript into polished text while preserving original meaning and voice.

CLEANING RULES:
• Fix capitalization, punctuation, spacing
• Remove filler words: "um", "uh", "like", "you know", "so", "actually"  
• Eliminate false starts and repetitions
• Correct speech recognition errors using context
• Fix homophones: their/there/they're, your/you're, its/it's
• Convert numbers appropriately (spell 1-9, numerals 10+)
• Break into logical paragraphs, maintain conversational tone
• Any paragraph cannot have more than 250 words.

CONTEXTUAL CORRECTIONS & ENTITY NORMALIZATION:
• When context strongly indicates a misrecognized proper noun (brands, companies, products, people) or number, correct it to the canonical form.
• Prioritize domain-appropriate forms and commonly used brand stylings.
• Examples:
  – If the content refers to the finance brand "360 ONE" but the transcript says "361", correct to "360 ONE".
  – If "G Pay" is recognized as "JPay" in a payments context, correct to "Google Pay" or "GPay" based on surrounding context.

AD/SPONSOR DETECTION:
• Identify obvious advertisements, sponsorships, or promotional content.
• Include: "This episode is sponsored by...", product placements, discount codes, promotional segments
• Wrap detected ads with [AD] tags: [AD]This episode is sponsored by BetterHelp. BetterHelp offers...[/AD]
• An Ad always deserves its own paragraph. If the guy keeps talking about the same thing, it's an ad.
• Don't flag brief organic mentions of companies in conversation context

CONSTRAINTS:
• Preserve original meaning. You may add/modify/delete words only to fix recognition errors, disfluencies, and formatting. Do not invent new facts or claims.

Return the cleaned output as a list of strings, where each string is a logical paragraph.

Process this transcript:"""

comprehensive_summary_prompt = f"""STEP 1: Identify the CORE THEME of this content in one clear sentence.

Then extract:

1. SUMMARY: Create a well-structured HTML overview
   - {HTML_FORMAT_RULES}
   - Lead with what's most interesting, group related ideas
   - Keep tone conversational and engaging

2. INSIGHTS: Self-contained factual statements aligned with core theme
   
   ABSOLUTE REQUIREMENTS - NO EXCEPTIONS:
   
   Each insight MUST be readable IN COMPLETE ISOLATION. Assume reader has read NOTHING else.
   
   FORBIDDEN - NEVER use:
   - "The software" / "This initiative" / "The company" / "The program" (without naming it EVERY TIME)
   - "He" / "She" / "They" (without stating WHO in that same sentence)
   - Acronyms without defining them IN THAT INSIGHT
   - "As mentioned" / "As discussed" / "The aforementioned"
   - ANY reference to other insights or earlier context
   
   REQUIRED in EVERY SINGLE INSIGHT:
   - Full proper names (people, companies, products, initiatives)
   - What each entity IS/DOES (define it)
   - All numbers with units (crores/dollars/months/people)
   - Specific timeframes (dates, durations, "since X")
   - Complete context (why this matters)
   
   MERGE related facts into ONE comprehensive insight. Don't split connected information. There is no limit on the number of insights you can extract.
   There is no limit on the number of characters in an insight.
   
   Test: Hand this insight to someone who knows NOTHING about the topic. Can they understand it fully? If no, REWRITE.
   
   Business strategies, opinions, industry analysis = INSIGHTS (not tangents)

3. TANGENTS: Truly off-topic moments unrelated to core theme
   - ONLY: personal stories, random anecdotes, unrelated trivia
   - Must ALSO be fully self-contained with complete context
   - Anything related to the main business/topic = INSIGHT, NOT tangent
   
   RULE: Related to core theme = INSIGHT. Random personal moment = TANGENT.

Return all three elements in the structured format.

Transcripted content:"""

MUNSHI_SYSTEM_PROMPT = """
Role: You are Munshi, an expert Communications Analyst. Your goal is to synthesize audio-derived text (podcasts, corporate meetings, or call recordings) into actionable insights, providing natural, high-level commentary as if you were a participant or a dedicated observer.

Tone & Style: * Professional yet Conversational: Avoid robotic "The text says" phrasing. Instead, use: "The discussion centers on...", "A recurring theme here is...", or "The participants seem to align on..."
"""

rag_answer_prompt = """
Instructions:
1. Answer the query directly and comprehensively.
2. Cite your sources naturally. When you reference specific information, include a citation in the format <a href="/output/{{UID}}">{{Title}}</a>. The Title and UID are provided in the chunk metadata.
3. Format your response in HTML. Use only standard HTML tags like <p>, <ul>, <li>, <strong>, <i>, and <a href="...">. Do not use Markdown.
4. Context-Aware: Subtly adapt your vocabulary to the format (e.g., using "Action Items" for meetings vs. "Narrative Arcs" for podcasts).
5. Analytical Wit: Provide "between the lines" observations. highlight the why behind the conversation.

WRITING STYLE - CRITICAL:
- Write like you're briefing someone who needs to know this
- NO corporate fluff: "synergies", "stakeholders", "value proposition", "learnings"
- BE SPECIFIC: Names, numbers, actual claims
- BE DIRECT: "X contradicts what they said in doc 2" not "different perspectives emerged"
- MAKE IT ACTIONABLE: Why should someone care about this?
- STRICTLY FORBIDDEN: Do NOT use phrases like "based on the provided chunks", "according to the transcript", "in the provided text", "the context suggests", or similar meta-references. Never break character or mention that you are processing text chunks.

Synthesize the provided context to answer the user's query with directness and depth. If the transcripts does not contain the specific information requested, clearly state that the recorded discussion does not appear to address that topic, rather than speculating.

Context:
{context}

Query:
{query}

Answer:"""

card_generation_prompt = """You are analyzing insights from multiple transcripts to generate an INSIGHT CARD.

SEED INSIGHT:
{seed_insight}

SIMILAR INSIGHTS FOUND:
{similar_count} insights with average similarity of {avg_similarity:.2f}

YOUR TASK:
1. Analyze the pattern in these insights
2. Determine the card type:
   - ECHO: Multiple independent sources confirming the same fact/truth
   - BRIDGE: Insights connecting two different topics/domains through a common pattern
   - FRACTURE: Multiple signals pointing to a problem, concern, or recurring issue
3. Generate structured card content

CARD TYPE DETECTION GUIDE:
- ECHO: "Everyone is saying X" - focus on convergence and validation
- BRIDGE: "X connects to Y" - focus on unexpected relationships between domains
- FRACTURE: "Many signals of problem Z" - focus on identifying issues and patterns

Return JSON with this EXACT structure:

{{
    "type": "echo" | "bridge" | "fracture",
    "title": "...",  // Compelling title (5-10 words)
    
    // FOR ECHO:
    "core_fact": "...",  // What everyone agrees on (if Echo)
    "significance": "...",  // Why convergence matters (if Echo)
    "convergence_score": 0.XX,  // Use average similarity (if Echo)
    
    // FOR BRIDGE:
    "connecting_insight": "...",  // Core insight that bridges (if Bridge)
    "cluster_a": "...",  // First domain/topic (if Bridge)
    "cluster_b": "...",  // Second domain/topic (if Bridge)
    "pattern_explanation": "...",  // Why connection matters (if Bridge)
    
    // FOR FRACTURE:
    "issue": "...",  // Core problem statement (if Fracture)
    "severity": "low" | "medium" | "high",  // Based on language/frequency (if Fracture)
    "top_patterns": ["...", "..."],  // 3-5 recurring themes (if Fracture)
    "recommendation": "..." or null  // Suggested action (if Fracture)
}}

CRITICAL: Choose ONE card type and populate only the fields for that type. Leave other type's fields out.

Context data:"""

