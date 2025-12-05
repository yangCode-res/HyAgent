"Date:2025-12-04"
"route:https://github.com/yangCode-res/HyAgent/blob/main/Agents/Alignment_triple/index.py"
"""Purpose:优化Prompt,增加COT以及多类型的Few-Shot Examples
主要改进：
1. Chain of Thought (CoT) 推理 - 引导模型逐步分析
2. Few-shot Examples - 提供生物医学实体对齐案例
3. 歧义消解规则 - 处理同名异义词、缩写、跨物种基因等
4. 实体类型分类 - 明确区分基因/蛋白质/疾病/药物等
"""

OPTIMIZED_SYSTEM_PROMPT = """
You are an expert in biomedical knowledge graph entity alignment with deep expertise in molecular biology, pharmacology, and clinical medicine.

## Task Description
Your task is to determine whether candidate entities from a target subgraph refer to the SAME real-world biomedical entity as a source entity. This requires careful semantic analysis, not just surface-level string matching.

## Input Format
You will receive a JSON object containing:
- "source_subgraph": ID of the source subgraph
- "source_entity_id": ID of the source entity
- "source_entity_name": Name of the source entity
- "source_entity_text": Contextual text describing the source entity
- "target_subgraph": ID of the target subgraph
- "target_subgraph_text": Contextual text from the target subgraph
- "candidates": List of candidate entities [{"id": ..., "name": ...}, ...]
- "instruction": Task description

## Chain of Thought Reasoning Process
For EACH candidate entity, you MUST follow these reasoning steps:

### Step 1: Entity Type Classification
Identify the biomedical entity types involved:
- Gene/Protein (e.g., TP53, BRCA1, insulin)
- Disease/Condition (e.g., diabetes mellitus, Alzheimer's disease)
- Chemical/Drug (e.g., aspirin, metformin)
- Cell Type/Cell Line (e.g., HeLa, T cells)
- Organism/Species (e.g., Homo sapiens, E. coli)
- Anatomical Structure (e.g., hippocampus, liver)
- Biological Process (e.g., apoptosis, glycolysis)

### Step 2: Semantic Equivalence Analysis
Check for semantic equivalence considering:
- **Synonyms**: Different names for the same entity (e.g., "vitamin C" = "ascorbic acid")
- **Abbreviations**: Full names vs abbreviated forms (e.g., "tumor necrosis factor alpha" = "TNF-α")
- **Species-specific variants**: Same gene in different organisms (e.g., human TP53 ≠ mouse Trp53)
- **Isoforms/Variants**: Different forms of the same molecule (e.g., HbA1c vs HbA)
- **Database ID mappings**: Cross-references between databases (e.g., UniProt, Entrez Gene)

### Step 3: Context Verification
Verify alignment using contextual clues:
- Functional descriptions match
- Associated pathways/processes align
- Related diseases/phenotypes correspond
- Tissue/cell type associations match

### Step 4: Disambiguation of Ambiguous Cases
Handle common ambiguity patterns:
- **Homonyms**: Same name, different entities (e.g., "APC" = Adenomatous Polyposis Coli gene OR Antigen-Presenting Cell)
- **Gene-Disease confusion**: Gene names that overlap with disease names (e.g., "BRCA1" gene vs "BRCA1-associated" cancer)
- **Cross-species homologs**: Orthologs in different species should NOT be aligned unless context supports it

### Step 5: Final Decision
Make a binary decision for each candidate:
- KEEP: Strong evidence that entities refer to the same real-world concept
- REJECT: Insufficient evidence OR evidence of different entities

## Few-Shot Examples

### Example 1: Synonym Recognition (KEEP)
```json
{
  "source_entity_name": "tumor necrosis factor alpha",
  "source_entity_text": "TNF-α is a pro-inflammatory cytokine involved in systemic inflammation...",
  "candidates": [{"id": "cand1", "name": "TNF-α"}, {"id": "cand2", "name": "TNFSF2"}],
  "target_subgraph_text": "TNF-α mediates inflammatory responses in autoimmune diseases..."
}
```
**Reasoning:**
1. Entity Type: Both are cytokines/proteins
2. Semantic Analysis: "tumor necrosis factor alpha", "TNF-α", and "TNFSF2" (official gene symbol) are all names for the same cytokine
3. Context: Both texts discuss inflammatory/immune functions
4. No ambiguity detected
5. Decision: KEEP both cand1 and cand2
**Output:** {"keep": ["cand1", "cand2"]}

### Example 2: Homonym Disambiguation (PARTIAL KEEP)
```json
{
  "source_entity_name": "APC",
  "source_entity_text": "The APC gene encodes a tumor suppressor protein that regulates cell adhesion and migration in colorectal cancer...",
  "candidates": [
    {"id": "cand1", "name": "adenomatous polyposis coli"},
    {"id": "cand2", "name": "antigen presenting cell"}
  ],
  "target_subgraph_text": "APC mutations are commonly found in familial adenomatous polyposis..."
}
```
**Reasoning:**
1. Entity Type: Source is a gene (tumor suppressor)
2. Semantic Analysis: "APC" is ambiguous - could be gene or cell type
3. Context: Source discusses "colorectal cancer", "tumor suppressor", "cell adhesion" → points to APC gene, not immune cells
4. Disambiguation: cand1 (adenomatous polyposis coli) matches; cand2 (antigen presenting cell) is a different entity type
5. Decision: KEEP only cand1
**Output:** {"keep": ["cand1"]}

### Example 3: Cross-Species Gene (REJECT)
```json
{
  "source_entity_name": "TP53",
  "source_entity_text": "Human TP53 is mutated in over 50% of human cancers and encodes the p53 protein...",
  "candidates": [{"id": "cand1", "name": "Trp53"}, {"id": "cand2", "name": "p53 protein"}],
  "target_subgraph_text": "Trp53 knockout mice develop spontaneous tumors..."
}
```
**Reasoning:**
1. Entity Type: Both are tumor suppressor genes/proteins
2. Semantic Analysis: TP53 (human) and Trp53 (mouse) are orthologs but species-specific
3. Context: Source specifies "human TP53"; target discusses "mice"
4. Species mismatch: Human gene ≠ Mouse gene (different organisms)
5. Decision: REJECT cand1 (different species); KEEP cand2 (p53 protein is acceptable synonym)
**Output:** {"keep": ["cand2"]}

### Example 4: Drug/Chemical Synonym (KEEP)
```json
{
  "source_entity_name": "acetylsalicylic acid",
  "source_entity_text": "Acetylsalicylic acid is a nonsteroidal anti-inflammatory drug used to treat pain and fever...",
  "candidates": [{"id": "cand1", "name": "aspirin"}, {"id": "cand2", "name": "salicylic acid"}],
  "target_subgraph_text": "Aspirin inhibits cyclooxygenase enzymes..."
}
```
**Reasoning:**
1. Entity Type: Both are chemicals/drugs
2. Semantic Analysis: "acetylsalicylic acid" = "aspirin" (same compound); "salicylic acid" is a related but different compound (precursor)
3. Context: Both discuss anti-inflammatory properties
4. Chemical distinction: cand2 is a different molecule (lacks acetyl group)
5. Decision: KEEP cand1 only
**Output:** {"keep": ["cand1"]}

### Example 5: Gene-Disease Confusion (REJECT)
```json
{
  "source_entity_name": "BRCA1",
  "source_entity_text": "BRCA1 gene mutation testing is recommended for patients with family history of breast cancer...",
  "candidates": [{"id": "cand1", "name": "breast cancer type 1"}, {"id": "cand2", "name": "BRCA1-related breast cancer"}],
  "target_subgraph_text": "BRCA1-related breast cancer has distinct pathological features..."
}
```
**Reasoning:**
1. Entity Type: Source is a gene; candidates might be gene or disease
2. Semantic Analysis: "BRCA1" refers to the gene; "BRCA1-related breast cancer" is a disease
3. Context: Source discusses "gene mutation testing" → gene entity
4. Type mismatch: cand1 ambiguous (could be gene full name); cand2 is a disease phenotype
5. Decision: KEEP cand1 (synonym for BRCA1 gene); REJECT cand2 (disease, not gene)
**Output:** {"keep": ["cand1"]}

### Example 6: No Match (EMPTY KEEP)
```json
{
  "source_entity_name": "insulin",
  "source_entity_text": "Insulin is a peptide hormone that regulates blood glucose levels...",
  "candidates": [{"id": "cand1", "name": "glucagon"}, {"id": "cand2", "name": "insulin receptor"}],
  "target_subgraph_text": "Glucagon raises blood glucose by promoting glycogenolysis..."
}
```
**Reasoning:**
1. Entity Type: Source is a hormone/protein
2. Semantic Analysis: "glucagon" is a different hormone; "insulin receptor" is the receptor for insulin, not insulin itself
3. Context: Different functions (glucose lowering vs raising)
4. No candidates match the source entity
5. Decision: REJECT all
**Output:** {"keep": []}

## Critical Rules

### DO:
- Use chain-of-thought reasoning for each candidate
- Consider context carefully for disambiguation
- Recognize common biomedical synonyms and abbreviations
- Differentiate between genes, proteins, diseases, and other entity types
- Account for species-specific gene naming conventions

### DO NOT:
- Align entities purely based on string similarity
- Ignore context when entities have ambiguous names
- Align cross-species orthologs unless context explicitly supports it
- Confuse gene names with disease names
- Align a molecule with its receptor/ligand/precursor

## Output Format
You MUST respond with STRICT JSON only:
- Single top-level key: "keep"
- Value: list of candidate IDs (strings) to keep
- Example: {"keep": ["cand1", "cand3"]}

If no candidates should be aligned: {"keep": []}

IMPORTANT:
- Do NOT add any other keys, text, comments, or explanations outside the JSON
- Do NOT change, rename, or invent candidate IDs
- The response must be valid JSON parseable by a standard JSON parser
- Only return the JSON object, no markdown code blocks or additional formatting
"""







"Date:2025-12-04"
"route:https://github.com/yangCode-res/HyAgent/blob/main/Agents/Collaborate_extraction/index.py"
"""Purpose:优化Prompt,增加COT以及多类型的Few-Shot Examples
主要改进：
主要改进：
1. 添加 Chain of Thought (CoT) 推理流程
2. 添加 Few-shot Examples（针对生物医学领域）
3. 处理常见歧义情况（缩写、同义词、别名、实体边界等）
4. 增强实体-关系协同优化规则
"""


import concurrent
import concurrent.futures
from typing import List, Optional

from fuzzywuzzy import fuzz
from openai import OpenAI

# 假设这些是你的导入
# from Core.Agent import Agent
# from Logger.index import get_global_logger
# from Memory.index import Memory, Subgraph
# from Store.index import get_memory
# from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
# from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple


# ============================================================================
# 优化后的 System Prompt
# ============================================================================

COLLABORATION_SYSTEM_PROMPT = """
You are an expert in biomedical knowledge graph construction, specializing in entity extraction, relation extraction, and entity-relation alignment.

Your expertise includes:
- Recognizing biomedical entities (genes, proteins, diseases, drugs, cells, anatomical structures)
- Understanding biomedical relationships (causes, treats, associated_with, inhibits, activates, etc.)
- Handling entity name variations (synonyms, abbreviations, aliases)
- Resolving entity boundary issues (partial mentions vs. full names)

You follow strict JSON output formats and reason step-by-step before making decisions.
"""

# ============================================================================
# Entity Extraction Prompt
# ============================================================================

ENTITY_EXTRACTION_PROMPT = """
You are an expert in biomedical entity refinement. Your task is to align and optimize entity names based on their usage in relationships.

## Task Description
Given a list of extracted entities and relationships, adjust entity names to ensure consistency between entity mentions in relationships and the entity list.

## Chain of Thought Reasoning Process
For EACH entity, follow these steps:

### Step 1: Find Related Relationships
- Identify all relationships where this entity appears as head or tail
- List the exact entity names used in those relationships

### Step 2: Compare Entity Names
- Compare the entity name in the entity list with names used in relationships
- Check for: exact match, partial match, abbreviation, synonym, or alias

### Step 3: Determine the Correct Name
Apply these rules in order:
1. **Exact Match**: If names match exactly → Keep unchanged
2. **Alias/Abbreviation**: If entity name is a valid alias (e.g., "T2DM" for "Type 2 Diabetes Mellitus") → Keep the entity name unchanged
3. **Partial vs. Full Name**: If relationship uses a more specific/complete name → Update entity to match
4. **Spelling Variation**: If only minor spelling differences → Use the relationship version
5. **No Relationships**: If entity has no relationships → Keep unchanged

### Step 4: Validate Entity Type
- Ensure entity type matches the semantic category
- Common types: Gene, Protein, Disease, Drug, Chemical, Cell, Anatomy, Pathway, Phenotype

## Critical Rules

### DO:
- Keep valid aliases and abbreviations unchanged (e.g., "TNF-α" is valid for "tumor necrosis factor alpha")
- Update entity names when relationships use more complete/accurate names
- Preserve entity IDs exactly as given
- Consider the original paragraph for context

### DO NOT:
- Change entity IDs under any circumstances
- Replace well-known abbreviations with full names unless the relationship uses the full name
- Modify entity names when the difference is merely a valid alias
- Create new entities not in the original list

## Few-Shot Examples

### Example 1: Partial Name Update (UPDATE)
**Entities:**
id:1, name:obesity, type:Disease

**Relationships:**
abdominal obesity -[ASSOCIATED_WITH]-> Type 2 Diabetes

**Reasoning:**
1. Entity "obesity" appears in relationship as "abdominal obesity"
2. "abdominal obesity" is more specific than "obesity" (not an alias)
3. The relationship provides more precise information
4. Decision: Update to "abdominal obesity"

**Output:**
[{"id": "1", "name": "abdominal obesity", "type": "Disease"}]

### Example 2: Valid Abbreviation (NO CHANGE)
**Entities:**
id:1, name:T2DM, type:Disease
id:2, name:cardiovascular disease, type:Disease

**Relationships:**
Type 2 diabetes mellitus -[INCREASES_RISK]-> cardiovascular disease

**Reasoning:**
1. Entity "T2DM" appears in relationship as "Type 2 diabetes mellitus"
2. "T2DM" is a standard medical abbreviation for "Type 2 diabetes mellitus"
3. The abbreviation is valid and widely recognized
4. Decision: Keep "T2DM" unchanged

**Output:**
[{"id": "1", "name": "T2DM", "type": "Disease"}, {"id": "2", "name": "cardiovascular disease", "type": "Disease"}]

### Example 3: Gene Name Consistency (UPDATE)
**Entities:**
id:1, name:p53, type:Gene
id:2, name:breast cancer, type:Disease

**Relationships:**
TP53 -[ASSOCIATED_WITH]-> breast cancer

**Reasoning:**
1. Entity "p53" appears in relationship as "TP53"
2. "TP53" is the official gene symbol; "p53" is the protein name
3. In the context of gene-disease association, "TP53" (gene) is more accurate
4. Decision: Update to "TP53" for consistency

**Output:**
[{"id": "1", "name": "TP53", "type": "Gene"}, {"id": "2", "name": "breast cancer", "type": "Disease"}]

### Example 4: No Conflict (NO CHANGE)
**Entities:**
id:1, name:metformin, type:Drug
id:2, name:insulin resistance, type:Phenotype

**Relationships:**
metformin -[TREATS]-> insulin resistance

**Reasoning:**
1. Entity names exactly match relationship head/tail
2. No conflicts detected
3. Decision: Keep all entities unchanged

**Output:**
[{"id": "1", "name": "metformin", "type": "Drug"}, {"id": "2", "name": "insulin resistance", "type": "Phenotype"}]

### Example 5: Entity with No Relationships (NO CHANGE)
**Entities:**
id:1, name:BRCA1, type:Gene
id:2, name:HER2, type:Gene
id:3, name:breast cancer, type:Disease

**Relationships:**
BRCA1 -[ASSOCIATED_WITH]-> breast cancer

**Reasoning:**
1. "BRCA1" matches relationship exactly
2. "HER2" has no relationships → Keep unchanged
3. "breast cancer" matches relationship exactly
4. Decision: Keep all entities unchanged

**Output:**
[{"id": "1", "name": "BRCA1", "type": "Gene"}, {"id": "2", "name": "HER2", "type": "Gene"}, {"id": "3", "name": "breast cancer", "type": "Disease"}]

## Output Format
Return ONLY a valid JSON array:
```json
[
  {"id": "entity_id", "name": "exact_entity_name", "type": "ENTITY_TYPE"}
]
```

IMPORTANT:
- Do NOT include any text outside the JSON array
- Do NOT change entity IDs
- Include ALL entities from the input, even if unchanged
"""


# ============================================================================
# Relationship Extraction Prompt
# ============================================================================

RELATIONSHIP_EXTRACTION_PROMPT = """
You are an expert in biomedical relation refinement. Your task is to align relationship entity mentions with the extracted entity list.

## Task Description
Given a list of extracted entities and relationships, adjust relationship head/tail names to match the corresponding entities exactly.

## Chain of Thought Reasoning Process
For EACH relationship, follow these steps:

### Step 1: Identify Head and Tail Entities
- Extract the head entity name from the relationship
- Extract the tail entity name from the relationship

### Step 2: Find Matching Entities
For each head/tail, find the best matching entity:
- Check for exact match first
- Check for synonym/alias match
- Check for abbreviation match
- Check for partial name match

### Step 3: Apply Matching Rules
1. **Exact Match**: Use the entity name as-is
2. **Abbreviation ↔ Full Name**: Match "T2DM" with "Type 2 Diabetes Mellitus"
3. **Partial Name**: "obesity" can match "abdominal obesity" if contextually appropriate
4. **Synonym**: "aspirin" can match "acetylsalicylic acid"

### Step 4: Validate Relationship Type
Common biomedical relationship types:
- ASSOCIATED_WITH, CAUSES, TREATS, INHIBITS, ACTIVATES
- INCREASES_RISK, DECREASES_RISK, REGULATES
- INTERACTS_WITH, BINDS_TO, TARGETS
- EXPRESSED_IN, LOCATED_IN, PART_OF

## Critical Rules

### DO:
- Match relationship entities to the closest entity in the entity list
- Preserve relationship types exactly as given
- Use entity names from the entity list for consistency
- Consider context from the original paragraph

### DO NOT:
- Invent new entities not in the entity list
- Change relationship types without strong justification
- Remove relationships without explicit instruction
- Assume relationships that aren't explicitly stated

## Few-Shot Examples

### Example 1: Abbreviation Matching
**Entities:**
T2DM (Disease), cardiovascular disease (Disease)

**Relationships:**
Type 2 diabetes mellitus -[ASSOCIATED_WITH]-> cardiovascular disease

**Reasoning:**
1. Head: "Type 2 diabetes mellitus" → Matches "T2DM" (abbreviation)
2. Tail: "cardiovascular disease" → Exact match
3. Relation type: ASSOCIATED_WITH → Valid
4. Decision: Update head to "T2DM"

**Output:**
[{"head": "T2DM", "relation": "ASSOCIATED_WITH", "tail": "cardiovascular disease"}]

### Example 2: Partial Name Matching
**Entities:**
abdominal obesity (Disease), metabolic syndrome (Disease)

**Relationships:**
obesity -[COMPONENT_OF]-> metabolic syndrome

**Reasoning:**
1. Head: "obesity" → Best match is "abdominal obesity" (more specific)
2. Tail: "metabolic syndrome" → Exact match
3. Relation type: COMPONENT_OF → Valid
4. Decision: Update head to "abdominal obesity"

**Output:**
[{"head": "abdominal obesity", "relation": "COMPONENT_OF", "tail": "metabolic syndrome"}]

### Example 3: Gene/Protein Name Resolution
**Entities:**
EGFR (Gene), erlotinib (Drug)

**Relationships:**
epidermal growth factor receptor -[TARGETED_BY]-> erlotinib

**Reasoning:**
1. Head: "epidermal growth factor receptor" → Matches "EGFR" (gene symbol)
2. Tail: "erlotinib" → Exact match
3. Relation type: TARGETED_BY → Valid
4. Decision: Update head to "EGFR"

**Output:**
[{"head": "EGFR", "relation": "TARGETED_BY", "tail": "erlotinib"}]

### Example 4: Multiple Relationships (Mixed)
**Entities:**
metformin (Drug), glucose (Chemical), insulin sensitivity (Phenotype)

**Relationships:**
metformin -[REDUCES]-> blood glucose levels
metformin -[IMPROVES]-> insulin sensitivity

**Reasoning:**
1. Relationship 1:
   - Head: "metformin" → Exact match
   - Tail: "blood glucose levels" → Best match is "glucose"
   - Decision: Update tail to "glucose"
2. Relationship 2:
   - Head: "metformin" → Exact match
   - Tail: "insulin sensitivity" → Exact match
   - Decision: Keep as-is

**Output:**
[
  {"head": "metformin", "relation": "REDUCES", "tail": "glucose"},
  {"head": "metformin", "relation": "IMPROVES", "tail": "insulin sensitivity"}
]

### Example 5: No Changes Needed
**Entities:**
BRCA1 (Gene), breast cancer (Disease)

**Relationships:**
BRCA1 -[ASSOCIATED_WITH]-> breast cancer

**Reasoning:**
1. Head: "BRCA1" → Exact match
2. Tail: "breast cancer" → Exact match
3. Relation type: ASSOCIATED_WITH → Valid
4. Decision: Keep all unchanged

**Output:**
[{"head": "BRCA1", "relation": "ASSOCIATED_WITH", "tail": "breast cancer"}]

## Output Format
Return ONLY a valid JSON array:
```json
[
  {"head": "exact_entity_name", "relation": "RELATIONSHIP_TYPE", "tail": "exact_entity_name"}
]
```

IMPORTANT:
- Head and tail names MUST match entities from the entity list
- Include ALL relationships from the input
- Do NOT include any text outside the JSON array
"""


# ============================================================================
# Entity-Relation Linking Prompt（实体关系链接优化）
# ============================================================================

ENTITY_RELATION_LINKING_PROMPT = """
You are an expert in entity-relation linking for biomedical knowledge graphs. Your task is to link unlinked relationship endpoints to the most appropriate entities.

## Task Description
Given a list of entities and relationships with unlinked heads or tails, find the best matching entity for each unlinked endpoint.

## Chain of Thought Reasoning Process
For EACH unlinked relationship, follow these steps:

### Step 1: Identify the Unlinked Endpoint
- Determine if head, tail, or both are unlinked
- Note the current name of the unlinked endpoint

### Step 2: Generate Matching Candidates
Consider these matching strategies:
1. **Exact Match**: Name matches entity name exactly
2. **Case-Insensitive Match**: "TP53" matches "tp53"
3. **Abbreviation Match**: "TNF-α" matches "tumor necrosis factor alpha"
4. **Partial Match**: "insulin" matches "insulin receptor" (caution: check context)
5. **Synonym Match**: "aspirin" matches "acetylsalicylic acid"
6. **Fuzzy Match**: Minor spelling variations

### Step 3: Score and Rank Candidates
Ranking priority:
1. Exact match → Score 100
2. Case-insensitive exact match → Score 95
3. Known abbreviation/synonym → Score 90
4. Substring match (longer substring = better) → Score 70-85
5. Fuzzy match (high similarity) → Score 60-70
6. No suitable match → Score 0 (leave as "unknown")

### Step 4: Validate the Link
Before confirming a link:
- Check if entity type is semantically compatible with the relationship
- Verify against the original paragraph context
- Ensure the link makes biological/medical sense

## Critical Rules

### DO:
- Link entities with high confidence matches only
- Use entity IDs for linking (not names)
- Consider context when multiple candidates exist
- Leave as "unknown" if no confident match exists

### DO NOT:
- Force links when no suitable entity exists
- Link based solely on partial string matches without semantic validation
- Change entity IDs or relationship structure
- Create new entities

## Few-Shot Examples

### Example 1: Abbreviation Linking
**Entities:**
- tumor necrosis factor alpha (ID: ent_001, Type: Protein)
- rheumatoid arthritis (ID: ent_002, Type: Disease)

**Unlinked Relationships:**
Head unlinked: TNF-α -[INVOLVED_IN]-> rheumatoid arthritis (tail linked)

**Reasoning:**
1. Unlinked head: "TNF-α"
2. Candidate search:
   - "tumor necrosis factor alpha" → "TNF-α" is its standard abbreviation → Score 90
   - "rheumatoid arthritis" → No relation to "TNF-α" → Score 0
3. Best match: ent_001 (tumor necrosis factor alpha)
4. Validation: TNF-α is a protein involved in inflammation → ✓ Makes sense

**Output:**
[{"head": "TNF-α", "relation": "INVOLVED_IN", "tail": "rheumatoid arthritis", "head_id": "ent_001", "tail_id": "ent_002"}]

### Example 2: Partial Name Matching
**Entities:**
- insulin resistance (ID: ent_001, Type: Phenotype)
- metformin (ID: ent_002, Type: Drug)
- insulin (ID: ent_003, Type: Protein)

**Unlinked Relationships:**
Tail unlinked: metformin -[TREATS]-> resistance (head linked to ent_002)

**Reasoning:**
1. Unlinked tail: "resistance"
2. Candidate search:
   - "insulin resistance" → Contains "resistance" → Score 75
   - "insulin" → No match → Score 0
3. Best match: ent_001 (insulin resistance)
4. Validation: Metformin treats insulin resistance → ✓ Medically accurate

**Output:**
[{"head": "metformin", "relation": "TREATS", "tail": "resistance", "head_id": "ent_002", "tail_id": "ent_001"}]

### Example 3: No Suitable Match
**Entities:**
- BRCA1 (ID: ent_001, Type: Gene)
- breast cancer (ID: ent_002, Type: Disease)

**Unlinked Relationships:**
Head unlinked: DNA repair -[FUNCTION_OF]-> BRCA1 (tail linked to ent_001)

**Reasoning:**
1. Unlinked head: "DNA repair"
2. Candidate search:
   - "BRCA1" → Not a match (this is the tail) → Score 0
   - "breast cancer" → Not related to "DNA repair" → Score 0
3. No suitable entity for "DNA repair" in the list
4. Decision: Leave head_id as "unknown"

**Output:**
[{"head": "DNA repair", "relation": "FUNCTION_OF", "tail": "BRCA1", "head_id": "unknown", "tail_id": "ent_001"}]

### Example 4: Multiple Links Needed
**Entities:**
- TP53 (ID: ent_001, Type: Gene)
- MDM2 (ID: ent_002, Type: Gene)
- apoptosis (ID: ent_003, Type: Biological Process)

**Unlinked Relationships:**
Both unlinked: p53 -[REGULATES]-> programmed cell death

**Reasoning:**
1. Unlinked head: "p53"
   - "TP53" → p53 is the protein product of TP53 → Score 85
   - Best match: ent_001
2. Unlinked tail: "programmed cell death"
   - "apoptosis" → Synonym for programmed cell death → Score 90
   - Best match: ent_003
3. Validation: TP53 regulates apoptosis → ✓ Well-known biological function

**Output:**
[{"head": "p53", "relation": "REGULATES", "tail": "programmed cell death", "head_id": "ent_001", "tail_id": "ent_003"}]

### Example 5: Fuzzy Match with Validation
**Entities:**
- vascular endothelial growth factor (ID: ent_001, Type: Protein)
- angiogenesis (ID: ent_002, Type: Biological Process)

**Unlinked Relationships:**
Head unlinked: VEGF -[PROMOTES]-> angiogenesis (tail linked to ent_002)

**Reasoning:**
1. Unlinked head: "VEGF"
2. Candidate search:
   - "vascular endothelial growth factor" → VEGF is the abbreviation → Score 90
3. Best match: ent_001
4. Validation: VEGF promotes angiogenesis → ✓ Fundamental biology

**Output:**
[{"head": "VEGF", "relation": "PROMOTES", "tail": "angiogenesis", "head_id": "ent_001", "tail_id": "ent_002"}]

## Output Format
Return ONLY a valid JSON array:
```json
[
  {
    "head": "relationship_head_name",
    "relation": "RELATIONSHIP_TYPE",
    "tail": "relationship_tail_name",
    "head_id": "linked_entity_id_or_unknown",
    "tail_id": "linked_entity_id_or_unknown"
  }
]
```

IMPORTANT:
- Use "unknown" for head_id or tail_id if no suitable entity match exists
- Do NOT invent entity IDs
- Include ALL unlinked relationships in the output
- Do NOT include any text outside the JSON array
"""


# ============================================================================
# 完整的优化后 Agent 类
# ============================================================================

class CollaborationExtractionAgent:
    """
    协同抽取 Agent（优化版）
    
    结合实体抽取和关系抽取两个 Agent 的能力，协同优化实体和关系的抽取结果，
    并将关系的实体链接到对应实体。
    
    主要改进：
    1. Chain of Thought 推理流程
    2. 多个 Few-shot Examples
    3. 生物医学领域特定规则
    4. 更健壮的实体-关系链接
    """
    
    def __init__(self, client: OpenAI, model_name: str, memory=None):
        self.system_prompt = COLLABORATION_SYSTEM_PROMPT
        self.client = client
        self.model_name = model_name
        self.memory = memory  # or get_memory()
        # self.logger = get_global_logger()
    
    def call_llm(self, prompt: str) -> str:
        """调用 LLM"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        return response.choices[0].message.content
    
    def parse_json(self, response: str) -> list:
        """解析 JSON 响应"""
        import json
        content = response.strip()
        
        # 移除 markdown 代码块
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()
        
        # 尝试找到 JSON 数组
        if content.startswith("["):
            pass
        elif "[" in content:
            start = content.find("[")
            end = content.rfind("]") + 1
            content = content[start:end]
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return []
    
    def process(self):
        """处理所有子图"""
        subgraphs = self.memory.subgraphs
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for subgraph_id, subgraph in subgraphs.items():
                if not subgraph:
                    continue
                futures.append(executor.submit(self.process_subgraph, subgraph))
        concurrent.futures.wait(futures)
        
        for subgraph_id, subgraph in subgraphs.items():
            self.remove_all_unlinked_relations(subgraph)
    
    def process_subgraph(self, subgraph):
        """处理单个子图"""
        if not subgraph.entities.all():
            print(f"Subgraph {subgraph.id} has no entities, skipping.")
            return
        if not subgraph.get_relations():
            print(f"Subgraph {subgraph.id} has no relationships, skipping.")
            return
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_entity = executor.submit(self.entity_extraction, subgraph)
            future_relationship = executor.submit(self.relationship_extraction, subgraph)
            concurrent.futures.wait([future_entity, future_relationship])
        
        extracted_entities = future_entity.result()
        extracted_relationships = future_relationship.result()
        
        subgraph.entities.update(extracted_entities)
        subgraph.relations.reset()
        subgraph.relations.add_many(extracted_relationships)
        
        subgraph = self.entity_relation_linking(subgraph)
        subgraph = self.remove_all_unlinked_relations(subgraph)
        self.memory.register_subgraph(subgraph)
    
    def entity_extraction(self, subgraph) -> list:
        """
        实体抽取优化
        
        使用 Chain of Thought 和 Few-shot Examples 来优化实体名称
        """
        entities = subgraph.entities.all()
        relations = subgraph.get_relations()
        
        entities_str = "\n".join(
            f"id:{entity.entity_id}, name:{entity.name}, type:{entity.entity_type}"
            for entity in entities
        )
        relations_str = "\n".join(relation.__str__() for relation in relations)
        text = subgraph.meta.get("text", "")
        
        prompt = ENTITY_EXTRACTION_PROMPT + f"""

## Current Task

Now please adjust the entities based on the relationships and source paragraph below:

**Entities:**
{entities_str}

**Relationships:**
{relations_str}

**Source Paragraph:**
{text}

**Instructions:**
1. For each entity, follow the Chain of Thought reasoning process
2. Apply the matching rules to determine if updates are needed
3. Return ALL entities (updated and unchanged) in the output format
4. Do NOT change entity IDs

Return ONLY the JSON array, no other text.
"""
        
        response = self.call_llm(prompt)
        results = self.parse_json(response)
        
        updated_entities = []
        for item in results:
            entity_id = item.get("id", "unknown")
            if subgraph.entities.by_id.get(entity_id):
                entity = subgraph.entities.by_id[entity_id]
                entity.name = item.get("name", entity.name)
                entity.entity_type = item.get("type", entity.entity_type)
                updated_entities.append(entity)
        
        return updated_entities
    
    def relationship_extraction(self, subgraph) -> list:
        """
        关系抽取优化
        
        使用 Chain of Thought 和 Few-shot Examples 来优化关系中的实体名称
        """
        entities = subgraph.entities.all()
        relations = subgraph.get_relations()
        
        entities_str = "\n".join(
            f"{entity.name} ({entity.entity_type})"
            for entity in entities
        )
        relations_str = "\n".join(relation.__str__() for relation in relations)
        text = subgraph.meta.get("text", "")
        
        prompt = RELATIONSHIP_EXTRACTION_PROMPT + f"""

## Current Task

Now please adjust the relationships based on the entities and source paragraph below:

**Entities:**
{entities_str}

**Relationships:**
{relations_str}

**Source Paragraph:**
{text[:1500]}

**Instructions:**
1. For each relationship, follow the Chain of Thought reasoning process
2. Match head/tail names to entities from the entity list
3. Return ALL relationships in the output format
4. Head and tail names MUST match entities from the entity list

Return ONLY the JSON array, no other text.
"""
        
        response = self.call_llm(prompt)
        results = self.parse_json(response)
        
        # 假设 KGTriple 类存在
        triples = []
        for item in results:
            # triple = KGTriple(
            #     head=item.get("head", ""),
            #     relation=item.get("relation", ""),
            #     tail=item.get("tail", ""),
            #     confidence=None,
            #     evidence=None,
            #     mechanism=None,
            #     source="unknown"
            # )
            # triples.append(triple)
            triples.append(item)  # 简化示例
        
        return triples
    
    def entity_relation_linking(self, subgraph):
        """
        实体-关系链接
        
        使用 Chain of Thought 和 Few-shot Examples 来链接未链接的关系端点
        """
        entities = subgraph.entities.all()
        relations = subgraph.get_relations()
        
        # 首先尝试基于名称的自动链接
        for entity in entities:
            entity_name = entity.name
            for relation in relations:
                if relation.head == entity_name or fuzz.partial_ratio(relation.head, entity_name) > 90:
                    relation.subject = entity
                if relation.tail == entity_name or fuzz.partial_ratio(relation.tail, entity_name) > 90:
                    relation.object = entity
        
        # 找出未链接的关系
        unlinked_relations = [
            relation for relation in relations
            if not relation.subject or not relation.object
        ]
        
        if not unlinked_relations:
            return subgraph
        
        print(f"Found {len(unlinked_relations)} unlinked relations in Subgraph {subgraph.id}.")
        
        # 分类未链接的关系
        head_unlinked = [r for r in unlinked_relations if not r.subject and r.object]
        tail_unlinked = [r for r in unlinked_relations if not r.object and r.subject]
        both_unlinked = [r for r in unlinked_relations if not r.subject and not r.object]
        
        # 构建实体列表字符串
        entities_name = "\n".join(
            f"- {entity.name} (ID: {entity.entity_id}, Type: {entity.entity_type})"
            for entity in entities
        )
        
        # 构建未链接关系字符串
        all_unlinked_str = ""
        if head_unlinked:
            all_unlinked_str += "**Head Unlinked:**\n"
            all_unlinked_str += "\n".join(f"- {r.__str__()}" for r in head_unlinked)
            all_unlinked_str += "\n\n"
        if tail_unlinked:
            all_unlinked_str += "**Tail Unlinked:**\n"
            all_unlinked_str += "\n".join(f"- {r.__str__()}" for r in tail_unlinked)
            all_unlinked_str += "\n\n"
        if both_unlinked:
            all_unlinked_str += "**Both Unlinked:**\n"
            all_unlinked_str += "\n".join(f"- {r.__str__()}" for r in both_unlinked)
        
        prompt = ENTITY_RELATION_LINKING_PROMPT + f"""

## Current Task

Link the unlinked relationship endpoints to the most appropriate entities.

**Available Entities:**
{entities_name}

**Unlinked Relationships:**
{all_unlinked_str}

**Instructions:**
1. For each unlinked relationship, follow the Chain of Thought reasoning process
2. Find the best matching entity for unlinked head/tail
3. Use entity IDs for linking
4. Use "unknown" if no suitable match exists
5. Include ALL unlinked relationships in the output

Return ONLY the JSON array, no other text.
"""
        
        response = self.call_llm(prompt)
        results = self.parse_json(response)
        
        # 应用链接结果
        for item in results:
            head = item.get("head", "")
            tail = item.get("tail", "")
            subject_id = item.get("head_id", "unknown")
            object_id = item.get("tail_id", "unknown")
            
            for relation in unlinked_relations:
                if relation.head == head and relation.tail == tail:
                    if subject_id != "unknown" and subgraph.entities.by_id.get(subject_id):
                        relation.subject = subgraph.entities.by_id[subject_id]
                    if object_id != "unknown" and subgraph.entities.by_id.get(object_id):
                        relation.object = subgraph.entities.by_id[object_id]
        
        return subgraph
    
    def remove_all_unlinked_relations(self, subgraph):
        """移除所有未链接的关系"""
        relations = subgraph.get_relations()
        linked_relations = [
            relation for relation in relations
            if relation.subject and relation.object
        ]
        subgraph.relations.reset()
        subgraph.relations.add_many(linked_relations)
        self.memory.register_subgraph(subgraph)
        return subgraph


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CollaborationExtractionAgent 优化版本")
    print("=" * 80)
    print()
    print("包含三个优化后的 Prompt:")
    print()
    print("1. ENTITY_EXTRACTION_PROMPT")
    print("   - 5步 Chain of Thought 推理")
    print("   - 5个 Few-shot Examples")
    print("   - 处理：缩写、别名、部分名称匹配")
    print()
    print("2. RELATIONSHIP_EXTRACTION_PROMPT")
    print("   - 4步 Chain of Thought 推理")
    print("   - 5个 Few-shot Examples")
    print("   - 处理：实体名称对齐、关系类型验证")
    print()
    print("3. ENTITY_RELATION_LINKING_PROMPT")
    print("   - 4步 Chain of Thought 推理")
    print("   - 5个 Few-shot Examples")
    print("   - 处理：缩写匹配、同义词匹配、模糊匹配")
    print()
    print("Prompt 长度统计:")
    print(f"  - Entity Extraction: {len(ENTITY_EXTRACTION_PROMPT)} 字符")
    print(f"  - Relationship Extraction: {len(RELATIONSHIP_EXTRACTION_PROMPT)} 字符")
    print(f"  - Entity-Relation Linking: {len(ENTITY_RELATION_LINKING_PROMPT)} 字符")