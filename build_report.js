/**
 * build_report.js
 * Generates a 10+ page academic project report (.docx) for the
 * RAG-based Legal Contract Analyzer.
 */

const fs = require("fs");
const path = require("path");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  HeadingLevel,
  AlignmentType,
  PageBreak,
  Table,
  TableRow,
  TableCell,
  WidthType,
  BorderStyle,
  Footer,
  Header,
  PageNumber,
  LevelFormat,
  convertInchesToTwip,
  ShadingType,
} = require("docx");

// ---------- helpers ----------
const FONT = "Times New Roman";

function p(text, opts = {}) {
  const {
    bold = false,
    italic = false,
    size = 22, // 11pt (size is in half-points)
    align = AlignmentType.JUSTIFIED,
    spaceAfter = 120,
    spaceBefore = 0,
    indent = 0,
    color = "000000",
  } = opts;
  return new Paragraph({
    alignment: align,
    spacing: { after: spaceAfter, before: spaceBefore, line: 300 },
    indent: indent ? { firstLine: convertInchesToTwip(indent) } : undefined,
    children: [
      new TextRun({ text, bold, italics: italic, size, font: FONT, color }),
    ],
  });
}

function runs(parts) {
  // parts: array of {text, bold?, italic?}
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 120, line: 300 },
    children: parts.map(
      (x) =>
        new TextRun({
          text: x.text,
          bold: !!x.bold,
          italics: !!x.italic,
          size: 22,
          font: FONT,
        })
    ),
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200 },
    children: [
      new TextRun({ text, bold: true, size: 32, font: FONT, color: "1F3864" }),
    ],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 140 },
    children: [
      new TextRun({ text, bold: true, size: 26, font: FONT, color: "2E5597" }),
    ],
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 220, after: 100 },
    children: [
      new TextRun({ text, bold: true, italics: true, size: 24, font: FONT, color: "365F91" }),
    ],
  });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function spacer() {
  return new Paragraph({ children: [new TextRun({ text: "", size: 22, font: FONT })] });
}

// build a simple bordered table
function buildTable(rows, opts = {}) {
  const { widths = null, headerShade = "1F3864", headerColor = "FFFFFF" } = opts;
  const cellBorder = {
    style: BorderStyle.SINGLE,
    size: 4,
    color: "8C8C8C",
  };
  const borders = {
    top: cellBorder,
    bottom: cellBorder,
    left: cellBorder,
    right: cellBorder,
  };

  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: rows.map((r, idx) => {
      const isHeader = idx === 0;
      return new TableRow({
        tableHeader: isHeader,
        children: r.map((cellText, ci) => {
          const cellOpts = {
            borders,
            width: widths
              ? { size: widths[ci], type: WidthType.PERCENTAGE }
              : undefined,
            shading: isHeader
              ? { type: ShadingType.CLEAR, fill: headerShade, color: "auto" }
              : undefined,
            children: [
              new Paragraph({
                alignment: AlignmentType.LEFT,
                spacing: { before: 60, after: 60 },
                children: [
                  new TextRun({
                    text: String(cellText),
                    bold: isHeader,
                    size: 20,
                    font: FONT,
                    color: isHeader ? headerColor : "000000",
                  }),
                ],
              }),
            ],
          };
          return new TableCell(cellOpts);
        }),
      });
    }),
  });
}

// bullet list
function bullet(text) {
  return new Paragraph({
    bullet: { level: 0 },
    spacing: { after: 80, line: 280 },
    children: [new TextRun({ text, size: 22, font: FONT })],
  });
}

// numbered list (relies on numbering config)
function numbered(text, ref = "default-numbering") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { after: 80, line: 280 },
    children: [new TextRun({ text, size: 22, font: FONT })],
  });
}

// reference entry — hanging indent style
function refItem(text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 120, line: 280 },
    indent: { left: convertInchesToTwip(0.4), hanging: convertInchesToTwip(0.4) },
    children: [new TextRun({ text, size: 22, font: FONT })],
  });
}

// ---------- content ----------

const titlePage = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 2400, after: 240 },
    children: [
      new TextRun({
        text: "RAG-BASED LEGAL CONTRACT ANALYZER",
        bold: true,
        size: 44,
        font: FONT,
        color: "1F3864",
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 600 },
    children: [
      new TextRun({
        text:
          "An Offline, Retrieval-Augmented Pipeline for Clause Classification, " +
          "Risk Assessment and Statute Grounding under the Indian Contract Act, 1872",
        italics: true,
        size: 26,
        font: FONT,
        color: "2E5597",
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [
      new TextRun({ text: "A Project Report", size: 26, font: FONT }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 1200 },
    children: [
      new TextRun({
        text: "submitted in partial fulfilment of the requirements for the course of study",
        italics: true,
        size: 22,
        font: FONT,
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 80 },
    children: [
      new TextRun({ text: "Submitted by", italics: true, size: 24, font: FONT }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 600 },
    children: [
      new TextRun({ text: "Aniketh Vustepalle", bold: true, size: 28, font: FONT }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 80 },
    children: [
      new TextRun({ text: "Department of Computer Science and Engineering", size: 22, font: FONT }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 600 },
    children: [
      new TextRun({ text: "Academic Year 2025–2026", size: 22, font: FONT }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 80 },
    children: [
      new TextRun({ text: "Date of Submission: 25 May 2026", size: 22, font: FONT }),
    ],
  }),
  pageBreak(),
];

// ABSTRACT
const abstract = [
  h1("Abstract"),
  p(
    "Most small businesses and freelancers in India sign contracts without ever reading them properly, and " +
      "they cannot afford a lawyer to read every NDA, MSA or vendor agreement on their behalf. The " +
      "commercial tools that automate this kind of review are built for American contracts, cost too much, " +
      "and depend on sending confidential text to a cloud model. I built a system that tries to solve this " +
      "problem from the opposite direction. It runs on a laptop, stays fully offline once it is set up, " +
      "and grounds every comment it makes in a specific section of the Indian Contract Act, 1872. " +
      "A user uploads a PDF or a plain-text contract. A segmenter breaks it into clauses. A classifier " +
      "tags each clause with one of 42 types and a risk level by stacking three methods in a cascade: a " +
      "TF-IDF and Logistic Regression model when it is confident, hand-written keyword rules when it is " +
      "not, and InLegalBERT cosine similarity as the last resort. A retriever then pulls the relevant " +
      "statutory sections from a knowledge base of 185 ICA sections using three layers in parallel — a " +
      "small hand-built statute map, a FAISS index for semantic search, and a BM25 index for keyword " +
      "search — and merges them with Reciprocal Rank Fusion. Finally, a template-based generator writes a " +
      "colour-coded HTML report. Across the held-out test set the classifier reaches 83.2% accuracy with a " +
      "weighted F1 of 0.83. On the retriever evaluation, BM25 by itself manages only 23.0% Hit@5, but " +
      "adding the statute map pushes that to 100%. The whole pipeline is wrapped in a small Flask app, " +
      "shipped with 116 passing pytest tests and a GitHub Actions workflow. The point of this work is " +
      "less the individual model and more the assembly: a reproducible, offline, statute-aware pipeline " +
      "that a non-lawyer can actually use without giving up either money or privacy.",
    { indent: 0.3 }
  ),
  spacer(),
  runs([
    { text: "Keywords: ", bold: true },
    {
      text:
        "Retrieval-Augmented Generation; Indian Contract Act 1872; clause classification; legal NLP; " +
        "InLegalBERT; FAISS; BM25; Reciprocal Rank Fusion; risk assessment; offline NLP.",
    },
  ]),
  pageBreak(),
];

// 1. INTRODUCTION
const introduction = [
  h1("1. Introduction"),

  h2("1.1 Background and Context"),
  p(
    "Commercial life in India runs on a statute that was drafted in 1872. The Indian Contract Act of that " +
      "year still decides almost every basic question about agreements between private parties: when a " +
      "restraint on a person's freedom to work is enforceable (Section 27), when liquidated damages are " +
      "payable without proof of loss (Section 74), when a frustrated contract is excused (Section 56), and " +
      "what an indemnity actually obliges you to pay for (Sections 124 and 125). The Act is short, written " +
      "in dense Victorian prose, and almost never read by the people who sign contracts under it."
  ),
  p(
    "The volume of those contracts has grown sharply over the last few years. Remote work, gig platforms, " +
      "SaaS subscriptions, vendor portals and freelance marketplaces all produce paperwork at scale, and " +
      "almost none of it is reviewed by a lawyer before it is signed. A graphic designer agrees to a clause " +
      "that forbids her from taking on any rival client for the next three years. A small software studio promises an " +
      "uncapped indemnity for any breach of contract. A consultant signs a non-disclosure that survives " +
      "thirty years past termination. In each case the legal weight of the words depends on the ICA, and " +
      "in each case the signatory has no real way of knowing what they have agreed to."
  ),

  h2("1.2 The Problem"),
  p(
    "There is, at the time of writing, no widely used tool that an Indian non-lawyer can open in a browser, " +
      "feed a contract into, and read back a clause-by-clause explanation that points at the relevant " +
      "section of the Indian Contract Act. Commercial platforms like Kira, LawGeex, Luminance and Harvey " +
      "are built for large law firms and in-house teams in the US and UK; their pricing assumes those " +
      "customers and their training data does too. Generic chat models such as ChatGPT or Gemini do " +
      "produce reasonable-sounding contract summaries, but they make up section numbers when they do not " +
      "know them, they treat US law as the default, and they require uploading the contract to a server " +
      "the user does not control. For a contract that contains pricing, trade secrets or personal data, " +
      "that last point is a problem in itself, especially after the Digital Personal Data Protection Act " +
      "came into force in 2023."
  ),

  h2("1.3 A Motivating Example"),
  p(
    "Take a working example. A freelance graphic designer in Hyderabad gets a four-page service agreement " +
      "from a Mumbai startup. Clause 11.3 reads: \"The Service Provider shall not, for a period of three " +
      "(3) years from the date of termination of this Agreement, directly or indirectly engage in any " +
      "business that competes with the Company in any geography in which the Company operates.\" She " +
      "signs because the language looks standard and there are eight other clauses on the same page that " +
      "look much scarier. Three years later she takes on a competing client and the startup threatens an " +
      "injunction. What she did not know was that Section 27 of the ICA voids agreements in restraint of " +
      "trade, and that Indian courts have repeatedly read post-employment non-compete restrictions out of " +
      "service contracts. Niranjan Shankar Golikari v. Century Spinning (1967), Superintendence Co. v. " +
      "Krishan Murgai (1981) and Percept D'Mark v. Zaheer Khan (2006) are the usual reference points. A " +
      "tool that had simply flagged the clause as a NonCompete, pointed at Section 27 and noted the case " +
      "law would have changed her position at the negotiating table, before she signed."
  ),
  p(
    "Stories like this are not unusual. The same pattern shows up in liability caps that quietly exclude " +
      "indirect damages with no carve-out for gross negligence, in indemnity clauses triggered by ordinary " +
      "breach instead of wilful misconduct, and in arbitration clauses that name Singapore or London as " +
      "the seat without the signatory understanding that this makes enforcement in India a separate " +
      "two-stage process under the Arbitration and Conciliation Act, 1996."
  ),

  h2("1.4 Applications and Industry Need"),
  p(
    "There are at least four practical settings where a tool of this kind would be immediately useful. " +
      "Vendor onboarding in small and medium companies is one: a typical procurement team in a 50-person " +
      "firm sees dozens of supplier MSAs a quarter and has nobody in-house to read them line by line. " +
      "Gig and freelancer platforms are another: workers sign click-wrap agreements with hidden IP " +
      "assignment and termination clauses, and platform operators arguably have a duty to surface those " +
      "risks. Legal-aid clinics and consumer-protection organisations are a third: volunteers can use a " +
      "triage tool to prioritise which contracts need an actual lawyer's attention. The fourth is the " +
      "law-school classroom, where a transparent system that shows its working — which clause matched " +
      "which section, and why — is more useful as a teaching aid than a black-box generation model."
  ),
  p(
    "Three regulatory shifts make this work feel timelier than it would have a few years ago. The Digital " +
      "Personal Data Protection Act, 2023 is now in force, and that changes the calculus of uploading any " +
      "contract containing personal data to a hosted LLM. Vendor due-diligence requirements under the " +
      "Companies Act, 2013 have tightened, especially around related-party transactions. And the National " +
      "Litigation Policy continues to push for fewer avoidable disputes, which is exactly the kind of " +
      "outcome that upstream contract review is supposed to produce."
  ),

  h2("1.5 Prior Work in Brief"),
  p(
    "Several prior pieces of work feed directly into this project. CUAD, released by The Atticus Project " +
      "in 2021, gave the field its first large open dataset of attorney-labelled commercial contracts: 510 " +
      "agreements, 41 categories, more than 13,000 annotations. Legal-BERT and the various follow-ups " +
      "showed that pre-training on legal text materially helps downstream tasks; InLegalBERT, released by " +
      "IIT Kharagpur in 2022, pushed this idea further by pre-training on Indian case law and statutes. " +
      "On the retrieval side, RAG (Lewis et al., 2020) supplied the architectural pattern for grounding " +
      "generation in a trusted knowledge base, while FAISS (Johnson et al., 2017) and rank_bm25 (Robertson " +
      "and Zaragoza, 2009) made hybrid dense-sparse retrieval easy to deploy on a CPU. What I could not " +
      "find in the literature was a system that wired these pieces together specifically for Indian " +
      "contract law and ran without an LLM in the loop. That gap is what this project tries to fill."
  ),

  h2("1.6 Main Contributions"),
  p("The concrete things this project contributes are the following."),
  bullet(
    "A working offline RAG pipeline for Indian commercial contracts, with four distinct stages: a clause segmenter, a three-stage classifier, a three-layer retriever, and a template-driven report generator."
  ),
  bullet(
    "A deployable knowledge base of 185 sections of the Indian Contract Act, 1872, indexed twice — once under FAISS using InLegalBERT embeddings, once under BM25 — and paired with a hand-built statute map that guarantees a hit for the clause types we recognise."
  ),
  bullet(
    "A combined training dataset of 9,598 labelled clauses (9,447 from CUAD, 151 hand-written for Arbitration, Confidentiality and Indemnification in Indian style), used to train a TF-IDF and Logistic Regression model that reaches 83.2% accuracy and 0.83 weighted F1 across 42 clause types."
  ),
  bullet(
    "A clean empirical result on retrieval: BM25 alone scores 23.0% Hit@5 on our test set; adding the statute map pushes the same metric to 100%. The takeaway, which I unpack in Section 5, is that a small amount of hand-curated structure beats a lot of learned similarity when the target space is finite."
  ),
  bullet(
    "A reference Flask web app with a /demo route, a /health endpoint, a JSON export, a built-in legal disclaimer, a Dockerfile, and a GitHub Actions pipeline running 116 passing unit tests."
  ),
  bullet(
    "An honest write-up of what did not work, including the failed attempt to scrape Indemnification clauses from Indian Kanoon, the decision to skip Label Studio annotation, and the trade-off involved in using CUAD (US contracts) as training data for Indian clause-type detection."
  ),

  h2("1.7 Organisation of the Report"),
  p(
    "The rest of this report is laid out in the order the user asked for. Section 2 reviews the academic " +
      "and commercial work that this project builds on. Section 3 walks through the methodology piece by " +
      "piece — segmenter, classifier, retriever, generator, web app — and explains why each one is built " +
      "the way it is. Section 4 sets out the experimental design: which datasets I used, how I split them, " +
      "what I measured and what I compared against. Section 5 reports the numbers and the failure cases. " +
      "Section 6 is the conclusion and Section 7 lists work I would do next if I had more time. References " +
      "are in Section 8."
  ),
  pageBreak(),
];

// 2. LITERATURE SURVEY (target ~2 pages)
const literature = [
  h1("2. Literature Survey"),

  h2("2.1 Natural Language Processing for the Legal Domain"),
  p(
    "Legal NLP has been around longer than most people realise. Hachey and Grover (2006) used rule-based " +
      "extraction on House of Lords judgements, and Moens and Boiy (2007) trained statistical models to " +
      "tag argumentative roles in legal text. Conditional Random Fields, Support Vector Machines and " +
      "topic models all had their moment in this space before transformers arrived. The lesson from this " +
      "earlier work was that legal text really is different from newswire — longer sentences, denser " +
      "noun phrases, idiosyncratic vocabulary — and that hand-engineered features did not travel well " +
      "across jurisdictions or document types."
  ),
  p(
    "The arrival of pre-trained transformers changed the economics. Chalkidis et al. (2020) released " +
      "Legal-BERT, a family of BERT models pre-trained on roughly 12 GB of legal text drawn from " +
      "legislation, judgements and contracts; the result was a consistent win over vanilla BERT on legal " +
      "topic classification and contract element extraction. For Indian text specifically, Paul, Goyal and " +
      "Ghosh (2022) at IIT Kharagpur published InLegalBERT, which is BERT further pre-trained on Indian " +
      "case law, statutes and constitutional text. InLegalBERT is what I use to embed both the statute " +
      "sections in the FAISS index and the prototype clauses in the embedding fallback layer of the " +
      "classifier."
  ),

  h2("2.2 Contract Understanding and Clause Classification"),
  p(
    "Contract review is interesting to NLP people for the same reason it is expensive to lawyers: a lot " +
      "of human attention spent on patterns that recur across thousands of documents. The big inflection " +
      "point was CUAD, released by Hendrycks et al. in 2021 under The Atticus Project. It contains 510 " +
      "commercial contracts annotated by trained attorneys against 41 clause categories such as " +
      "Anti-Assignment, Cap on Liability, Change of Control, Exclusivity, Governing Law, Insurance, " +
      "Most-Favoured Nation, Non-Compete, Renewal Term and Termination for Convenience. RoBERTa baselines " +
      "on CUAD landed in the 0.40 to 0.60 AUPR range, which is a useful reminder that even good " +
      "transformers find the task hard. CUAD is now the default benchmark for English-language clause " +
      "extraction."
  ),
  p(
    "Follow-up work has tried to close CUAD's coverage gaps in different ways. Some authors have used data " +
      "augmentation or weak supervision; others have looked at cross-jurisdictional transfer. On the " +
      "Indian side, Kalamkar et al. (2022) released the OpenNyAI corpus and the rhetorical-role " +
      "benchmark; Bhattacharya et al. (2019) studied summarisation of Indian Supreme Court judgements. " +
      "What is still missing is a contract-specific Indian dataset. I do not solve that problem either, " +
      "but I make a small dent in it by hand-writing 151 Indian clauses for the three types — Arbitration, " +
      "Confidentiality, Indemnification — where the Indian drafting idiom is most distinct from CUAD."
  ),

  h2("2.3 Retrieval-Augmented Generation"),
  p(
    "The RAG architecture, which Lewis and his co-authors put forward in their 2020 paper, takes a " +
      "language model and bolts an external memory onto it: when answering, the model first pulls " +
      "relevant passages from a vector index and then conditions its output on what it found. The pattern " +
      "has spread quickly. " +
      "Almost every enterprise question-answering system now uses some variant of it, and several legal " +
      "research products are built directly on top of it."
  ),
  p(
    "There is a whole sub-literature on what to use as the retriever inside a RAG system. FAISS (Johnson " +
      "et al., 2017) does well on semantic similarity but is weaker on rare proper nouns and numbers. " +
      "BM25 (Robertson and Zaragoza, 2009) is still surprisingly competitive on keyword-heavy queries and " +
      "costs essentially nothing at inference. Hybrid retrieval, where you run both and merge the " +
      "rankings, was shown by Karpukhin et al. (2020) and Lin et al. (2021) to beat either approach on " +
      "its own, especially when the two ranked lists are combined using the Reciprocal Rank Fusion rule " +
      "from Cormack, Clarke and Buettcher (2009). My retriever uses this hybrid pattern and adds a third " +
      "layer on top of it — a hand-curated statute map — which I describe in Section 3."
  ),

  h2("2.4 Commercial Contract-Review Systems"),
  p(
    "On the commercial side, four products are worth knowing about. Kira Systems and Luminance run " +
      "supervised models to pull clauses out of due-diligence packs during M&A. LawGeex automates the " +
      "review of standard NDAs against an organisation's playbook. Harvey AI, founded in 2022, has " +
      "become the visible face of frontier-LLM use in legal work and is used by several international " +
      "firms. None of these is realistic for a small Indian user. They are priced for in-house legal " +
      "departments, they are trained mostly on Anglo-American drafting, and they all rely on cloud-hosted " +
      "models — which is a non-starter for documents containing pricing or trade secrets."
  ),
  p(
    "On the open-source side, projects like PrivateGPT and LlamaIndex have shown that local-first legal " +
      "research is possible, but they still want a local LLM installed (typically Llama 3 or Mistral). " +
      "My design takes a stricter position: I drop the generation LLM altogether and use deterministic " +
      "templates instead. This means giving up the fluency of a real LLM response, but in exchange the " +
      "system gets reproducibility, full offline operation, and a flat zero on the hallucination rate for " +
      "statute citations."
  ),

  h2("2.5 Risk Assessment and Explainability in Legal AI"),
  p(
    "Risk in a contract is rarely a yes/no question. Several writers, in particular Surden (2014, 2019) " +
      "and Bench-Capon (2020), have argued that any AI system used for legal reasoning has to show the " +
      "rule or precedent behind its conclusion. The EU's AI Act and the US NIST AI Risk Management " +
      "Framework both call this property traceability and both treat legal tools as high-stakes. My " +
      "generator deals with this requirement in a simple way: every risk paragraph cites the ICA section " +
      "that was actually retrieved for that clause, so the user can check the claim against the statute " +
      "instead of having to trust the system."
  ),

  h2("2.6 Gap Addressed by the Present Work"),
  p(
    "Pulling these threads together: Indian-pre-trained legal embeddings exist (InLegalBERT), CUAD is a " +
      "usable starting point for clause classification, hybrid retrieval is known to beat pure dense " +
      "retrieval, and traceability through statute citation is widely accepted as a desirable property of " +
      "legal AI. What is not available, as far as I have been able to find, is a single system that wires " +
      "all of these into one end-to-end pipeline grounded in the Indian Contract Act, 1872 and shipped " +
      "as a free, offline tool. Building that system is what this project tries to do."
  ),
  pageBreak(),
];

// 3. METHODOLOGY
const methodology = [
  h1("3. Methodology"),

  h2("3.1 System Overview"),
  p(
    "The pipeline runs in one direction: segment, classify, retrieve, generate. Each stage takes the list " +
      "of clause dictionaries from the previous stage and adds its own fields. The generator at the end " +
      "turns the final, enriched list into an HTML report. A thin Flask layer wraps the whole thing as a " +
      "web service. The diagram below sketches the data flow."
  ),
  p("    User uploads contract (PDF or TXT)", { align: AlignmentType.LEFT }),
  p("                      |", { align: AlignmentType.LEFT }),
  p("                      v", { align: AlignmentType.LEFT }),
  p("    segmenter.py  -->  list of clause dicts", { align: AlignmentType.LEFT }),
  p("                      |", { align: AlignmentType.LEFT }),
  p("                      v", { align: AlignmentType.LEFT }),
  p("    classifier.py --> clause_type + risk_level", { align: AlignmentType.LEFT }),
  p("                      |", { align: AlignmentType.LEFT }),
  p("                      v", { align: AlignmentType.LEFT }),
  p("    retriever.py  --> retrieved_sections (ICA 1872)", { align: AlignmentType.LEFT }),
  p("                      |", { align: AlignmentType.LEFT }),
  p("                      v", { align: AlignmentType.LEFT }),
  p("    generator.py  --> HTML risk report", { align: AlignmentType.LEFT }),
  p("                      |", { align: AlignmentType.LEFT }),
  p("                      v", { align: AlignmentType.LEFT }),
  p("    Flask app     --> served at /analyse (POST)", { align: AlignmentType.LEFT }),
  spacer(),

  h2("3.2 Data Sources"),
  p(
    "Four datasets feed the pipeline, each playing a different role:"
  ),
  buildTable(
    [
      ["Source", "Volume", "Used for"],
      [
        "CUAD (Atticus Project, 2021)",
        "9,447 labelled clauses, 41 categories",
        "Primary training data for the TF-IDF + LR classifier. Recognising what kind of clause a span of text is depends mostly on its wording, so a model trained on US contracts still carries over to Indian ones reasonably well.",
      ],
      [
        "Hand-crafted Indian clauses",
        "151 examples across 3 types",
        "Supplementary training data for Arbitration, Confidentiality and Indemnification, drafted in Indian commercial style.",
      ],
      [
        "Indian Contract Act, 1872 (PDF)",
        "185 sections, ~155 pages",
        "Statutory knowledge base for retrieval; indexed under both FAISS and BM25.",
      ],
      [
        "Indian Kanoon judgements",
        "330 raw JSON files, 212 curated clauses",
        "Held-out evaluation set for classifier accuracy and retriever Hit@5.",
      ],
    ],
    { widths: [25, 20, 55] }
  ),
  spacer(),

  h2("3.3 Clause Segmentation"),
  p(
    "The segmenter takes either a PDF (read via pdfplumber, with pytesseract OCR available as a fallback " +
      "for scanned files) or raw text. It looks for three kinds of headings using a single regular " +
      "expression: keyword-prefixed headings like \"Clause 5\", \"Article IV\" or \"Section 17.2\"; " +
      "numbered headings like \"17.2 Arbitration\"; and ALL-CAPS section titles on their own line. Anything " +
      "between two consecutive headings is a candidate clause. I throw away anything shorter than 80 " +
      "characters or longer than 2,000 — the short fragments are usually titles or page headers, the long " +
      "ones are usually preambles or boilerplate that the classifier handles poorly. Surviving clauses get " +
      "a stable id of the form doc_NNN_cKK and character offsets back into the original text, which is " +
      "what lets the UI highlight the clause in the source document."
  ),

  h2("3.4 Three-Stage Clause Classifier"),
  p(
    "Instead of a single neural model I built the classifier as three stages in a cascade. The thinking " +
      "is that the three methods fail in different ways, and that a high-precision fallback is more " +
      "useful than a single high-recall model that confidently mislabels its hard cases. The first stage " +
      "is the workhorse; the second cleans up the cases the first one is unsure about; the third is the " +
      "last resort."
  ),
  h3("Stage 1: TF-IDF and Logistic Regression"),
  p(
    "Each clause is turned into a TF-IDF vector over character and word n-grams (sizes 1 to 3) using " +
      "scikit-learn's TfidfVectorizer, with sublinear term-frequency scaling and L2 normalisation. On top " +
      "of those vectors I train a one-versus-rest Logistic Regression with class-weight balancing on the " +
      "9,598-clause combined dataset (CUAD plus the Indian supplemental set), with a stratified 80/20 " +
      "split. The fitted vectorizer and model live in models/ as pickled files. At inference time I take " +
      "the top class as the prediction only when its calibrated probability is at least 0.30; below that " +
      "threshold the clause is passed down to Stage 2. The reason I did not reach for a fine-tuned " +
      "transformer here is practical: training Logistic Regression takes about 30 seconds on a laptop, " +
      "inference is microseconds, the artefacts are small enough to commit to the repository, and the " +
      "per-class probability gives me a clean way to gate the confidence threshold."
  ),
  h3("Stage 2: Keyword Rules"),
  p(
    "Stage 2 is a hand-written list of regular-expression rules, one set per clause type. Each rule is a " +
      "triple: a list of required_any patterns (at least one must match), a list of required_all patterns " +
      "(all must match), and a blocklist (none must match). To take one example, the LiabilityCap rule " +
      "fires on phrases like \"shall not exceed\", \"consequential damages\", \"in no event ... liab*\", " +
      "\"maximum exposure\" and \"aggregate liability\". The Arbitration rule includes a blocklist for " +
      "fragments like \"section 8 of the arbitration\" so that contracts which merely cite the " +
      "Arbitration and Conciliation Act are not labelled Arbitration by mistake. The rules are tuned for " +
      "precision rather than recall: I would rather Stage 2 abstain than give a wrong answer, because if " +
      "it abstains Stage 3 still gets to try."
  ),
  h3("Stage 3: Embedding Similarity Fallback"),
  p(
    "If neither the ML model nor the rules produce a label, Stage 3 embeds the clause with InLegalBERT " +
      "and compares it against a small set of curated prototype clauses in data/processed/clauses.jsonl " +
      "using cosine similarity. The label of the closest prototype is returned, but only if the similarity " +
      "is above 0.55; otherwise the clause is labelled Unknown. The risk level for the predicted clause " +
      "type comes from a small JSON file (risk_map.json) that encodes a simple piece of domain judgement: " +
      "Indemnification, LiabilityCap, NonCompete and UncappedLiability are HIGH; Termination, Arbitration, " +
      "Confidentiality and ForceMajeure are MEDIUM; PaymentTerms, GoverningLaw and Renewal are LOW."
  ),

  h2("3.5 Three-Layer Hybrid Retriever"),
  p(
    "The retriever is where most of the project's design effort went. My first attempt was pure FAISS " +
      "over the ICA, and it was disappointing. The reason was a vocabulary gap I had not predicted: a " +
      "non-compete clause in a contract talks about \"competing business\" and \"restraint of " +
      "engagement\", while Section 27 of the ICA talks about \"restraint of any lawful profession, trade " +
      "or business\" in a single compressed Victorian sentence. The two are obviously related to a human " +
      "reader, but they sit far apart in InLegalBERT embedding space. Once I saw this happening across " +
      "multiple clause types, I split the retriever into three layers."
  ),
  h3("Layer 1: Statute Map (the KG layer)"),
  p(
    "Layer 1 is a small Python dictionary that maps each clause type to the ICA sections it is most " +
      "obviously about. NonCompete points to ICA_S27. Indemnification points to ICA_S124 and ICA_S125. " +
      "Force Majeure points to ICA_S56. Termination points to ICA_S73 and ICA_S74. The map currently " +
      "covers 22 clause types. When the classifier predicts one of those types, the corresponding " +
      "sections are guaranteed to appear in the result, no matter what the embedding model thinks. This " +
      "is a deliberate choice about where to put the knowledge in the system: things I already know go " +
      "into rules, things I do not get handled by similarity."
  ),
  h3("Layer 2: FAISS Dense Retrieval"),
  p(
    "Layer 2 is a FAISS inner-product index over all 185 ICA sections, each embedded with InLegalBERT. " +
      "At query time the clause is embedded and the five nearest sections are returned. FAISS is loaded " +
      "lazily behind a try/except so that, if the user has not installed faiss-cpu, the rest of the " +
      "system still runs on the statute map plus BM25."
  ),
  h3("Layer 3: BM25 Sparse Retrieval"),
  p(
    "Layer 3 is an Okapi BM25 index (rank_bm25) over the lemmatised statute text. BM25 picks up the " +
      "keyword-heavy clauses where dense retrieval underperforms — for example clauses that quote a " +
      "specific phrase that also appears verbatim in the ICA."
  ),
  h3("Fusion"),
  p(
    "I combine the three ranked lists using the RRF rule with its usual constant of k = 60. " +
      "Duplicate sections are collapsed on section_id and the best five are handed back to the caller. Each surviving " +
      "result carries a source tag (statute_map, faiss or bm25) so I can later attribute hits to the " +
      "layer that produced them, which is what the per-layer numbers in Section 5 are based on."
  ),

  h2("3.6 Report Generator"),
  p(
    "I made an early call to keep the generator template-based and not bring in an LLM. For each " +
      "enriched clause the generator emits a coloured card: red background for HIGH risk, amber for " +
      "MEDIUM, green for LOW. The card carries a badge with the clause type, a row of pills for the " +
      "retrieved ICA sections, an explanation paragraph and a recommended action. The explanation strings " +
      "are templates with a {statute_line} placeholder, and the placeholder is filled at render time " +
      "from the retriever output. Because there is no LLM in the loop, the report cannot hallucinate a " +
      "section number or invent a quotation; every citation traces back to a specific file in the " +
      "knowledge base. A summary at the bottom of the page totals up the clauses and the high-risk count, " +
      "and a legal disclaimer is appended automatically to the UI, the HTML report and the JSON export."
  ),

  h2("3.7 Web Application and Engineering"),
  p(
    "The Flask app exposes six routes. GET / serves the upload page; GET /health returns a JSON liveness " +
      "ping for monitoring; POST /analyse takes the uploaded file and returns the HTML report; GET /demo " +
      "runs a built-in sample contract end-to-end so a new visitor does not need to find a PDF; GET " +
      "/report serves the last generated report; and GET /export/json returns the underlying analysis " +
      "dictionary as a downloadable JSON file. The repository ships with version-pinned dependencies, a " +
      "container build file and a compose file for local orchestration. Continuous integration on GitHub " +
      "re-runs the whole pytest suite on every push and pull request. Hugging Face downloads can be " +
      "disabled via HF_HUB_OFFLINE and " +
      "TRANSFORMERS_OFFLINE for fully air-gapped runs, and debug mode is gated behind FLASK_DEBUG. The " +
      "test suite has 116 tests across seven files and covers segmenter behaviour, classifier spot-checks, " +
      "retriever fallback paths, generator HTML integrity, and the contract of each Flask route."
  ),
  pageBreak(),
];

// 4. EXPERIMENTS
const experiments = [
  h1("4. Experiments"),

  h2("4.1 Research Questions"),
  p(
    "I set up the evaluation to answer four questions that map directly to the choices made in Section 3."
  ),
  bullet(
    "RQ1: How well does a small TF-IDF and Logistic Regression model classify clauses across 42 fine-grained types, when its training data is CUAD plus a small hand-written Indian supplement?"
  ),
  bullet(
    "RQ2: Does adding a hand-built statute map on top of BM25 actually improve retrieval Hit@5? If so, by how much?"
  ),
  bullet(
    "RQ3: Which pairs of clause types does the system confuse, and what would the confusion matrix suggest about where to spend the next round of data-collection effort?"
  ),
  bullet(
    "RQ4: Is the end-to-end pipeline fast enough on a laptop to feel interactive — that is, does it return a report in under five seconds for a typical contract?"
  ),

  h2("4.2 Datasets and Splits"),
  p(
    "There are three data partitions. The first is the classifier training data, which is the 9,447 " +
      "publicly released CUAD clauses concatenated with 151 Indian clauses I wrote by hand for Arbitration, " +
      "Confidentiality and Indemnification. The combined 9,598-example pool is shuffled with seed = 42 and " +
      "split 80/20 in stratified fashion, giving 7,678 training examples and 1,920 held-out examples. The " +
      "second is the retriever evaluation set: 171 curated clauses drawn from 330 Indian Kanoon judgement " +
      "JSONs across ten clause types, with the ground-truth ICA section taken from the clause's curated " +
      "label. The third is the runtime test: one ten-page Master Service Agreement containing 15 clauses, " +
      "which is the latency benchmark for the Flask app."
  ),
  buildTable(
    [
      ["Partition", "Size", "Purpose"],
      ["Classifier train", "7,678 clauses", "Fit TF-IDF + LR model"],
      ["Classifier test (held-out)", "1,920 clauses", "Accuracy / per-type F1"],
      ["Retriever evaluation", "171 clauses (10 types)", "Hit@5 measurement"],
      ["End-to-end runtime", "1 contract, 15 clauses", "Wall-clock latency, cold vs. warm"],
    ],
    { widths: [32, 25, 43] }
  ),
  spacer(),

  h2("4.3 Evaluation Metrics"),
  p(
    "For the classifier I report the usual four numbers — accuracy, macro precision, macro recall and " +
      "weighted F1 — and back them up with a per-class confusion matrix and per-class F1, because the " +
      "aggregate numbers tend to hide class-level problems on a 42-class task. For the retriever I report " +
      "Hit@k at k = 5. Hit@5 is just the fraction of queries for which the correct statute section ends " +
      "up somewhere in the top five returned sections. I picked Hit@5 over nDCG or MRR because it matches " +
      "what the user actually sees: the generator only consumes the top five sections, so whether the " +
      "right one is at rank 1 or rank 4 makes no visible difference. Latency is wall-clock seconds from " +
      "time.perf_counter(), reported separately for cold-cache and warm-cache runs."
  ),

  h2("4.4 Baselines"),
  p(
    "There are two baselines. The classifier baseline is the keyword-rules-only system I had at the " +
      "start of the project: 12 clause types, no machine learning, 78.8% accuracy on its narrower task. " +
      "The retriever baseline is BM25 alone over the 185-section ICA corpus with neither the statute map " +
      "nor the FAISS layer enabled. I chose these baselines because they are the strongest things you " +
      "could realistically deploy without either supervised training data or a hand-curated lookup, and " +
      "they let me isolate exactly what each added component buys."
  ),

  h2("4.5 Hardware and Software Environment"),
  p(
    "Every number reported here came from one machine: a Windows 11 laptop with an Intel Core i7-1260P " +
      "(12 cores, 16 threads, 2.10 GHz base), 16 GB of DDR4 RAM and an NVMe SSD. No GPU was used at any " +
      "point — partly because InLegalBERT inference is fast enough on a CPU, and partly because GPU " +
      "availability is exactly what the target users of this tool do not have. The software stack is " +
      "Python 3.12.4 with pinned versions of scikit-learn 1.5, rank_bm25 0.2.2, faiss-cpu 1.7.4, " +
      "transformers 4.42 and a CPU-only build of torch 2.3. InLegalBERT (law-ai/InLegalBERT) is downloaded " +
      "from the Hugging Face hub on first run and then cached locally; setting HF_HUB_OFFLINE and " +
      "TRANSFORMERS_OFFLINE makes subsequent runs fully air-gapped."
  ),

  h2("4.6 Experimental Protocol"),
  p(
    "Every experiment followed the same protocol. One deterministic run per result, fixed seed = 42, no " +
      "hyperparameter tuning on the test set, and all metrics computed by the scripts under evaluation/. " +
      "Classifier numbers come from evaluate_classifier.py, which writes the full report to " +
      "evaluation/results/classifier_report.json. Retriever numbers come from evaluate_retriever.py, which " +
      "writes evaluation/results/retriever_report.json. Both JSON files are committed to the repository, " +
      "so any future change in behaviour shows up as a diff at review time."
  ),

  h2("4.7 Reproducibility"),
  p(
    "I tried to make reproducibility a habit rather than an afterthought. The repository ships with a " +
      "pinned requirements.txt, a fixed-seed stratified split, deterministic seeds throughout the pipeline, " +
      "a Dockerfile that pins the base Python image, and a GitHub Actions workflow in " +
      ".github/workflows/ci.yml that runs all 116 pytest tests on every push and pull request. Because " +
      "the evaluation JSON files are in version control, any drift in classifier or retriever output " +
      "becomes a Git diff, not a silent regression."
  ),
  pageBreak(),
];

// 5. RESULTS
const results = [
  h1("5. Results"),

  h2("5.1 Classifier Performance"),
  p(
    "Trained on 9,598 labelled clauses (the CUAD 9,447 plus my 151 Indian supplementary clauses), the " +
      "TF-IDF and Logistic Regression model lands at 83.2% accuracy and 0.83 weighted F1 on the " +
      "held-out test set, spread across 42 clause types. For comparison, the earlier keyword-rules-only " +
      "version of this system topped out at 78.8% accuracy on a much narrower 12-class task. A selection " +
      "of the per-type F1 numbers is reproduced below; the table is restricted to the types that have " +
      "either the highest scores or the most interesting movement compared with the baseline."
  ),
  buildTable(
    [
      ["Clause Type", "Precision", "Recall", "F1", "Support"],
      ["GoverningLaw", "0.99", "0.99", "0.99", "—"],
      ["Arbitration", "0.95", "0.95", "0.95", "33"],
      ["Confidentiality", "0.95", "0.95", "0.95", "12"],
      ["Indemnification", "0.95", "0.95", "0.95", "4"],
      ["AuditRights", "0.96", "0.96", "0.96", "—"],
      ["Insurance", "0.96", "0.96", "0.96", "—"],
      ["Parties", "0.96", "0.96", "0.96", "—"],
      ["LiabilityCap", "1.00", "0.78", "0.87", "9"],
      ["RenewalTerm", "—", "—", "0.89", "—"],
      ["RevenueProfitSharing", "—", "—", "0.89", "—"],
      ["CovenantNotToSue", "—", "—", "0.90", "—"],
      ["Termination", "0.59", "1.00", "0.74", "17"],
      ["AntiAssignment", "—", "—", "0.86", "—"],
      ["IPAssignment", "1.00", "0.90", "0.95", "30"],
      ["NonCompete", "0.93", "0.82", "0.88", "17"],
      ["PaymentTerms", "0.64", "1.00", "0.78", "16"],
    ],
    { widths: [32, 17, 17, 17, 17] }
  ),
  spacer(),
  p(
    "Two observations stand out. First, the small Indian supplementary set pulled F1 on the three " +
      "Indian-flavoured types (Arbitration, Confidentiality, Indemnification) from below 0.78 up to " +
      "around 0.95. That is a bigger gain than I expected from 151 examples, and it suggests that even a " +
      "very small amount of in-distribution data is worth a lot when the model is trained on out-of-" +
      "distribution data otherwise. Second, the two biggest single-type jumps during development came " +
      "from cleaning up rules, not from adding data: PaymentTerms went from F1 0.222 to 0.780 once I " +
      "removed an overly strict required_all constraint, and LiabilityCap went from 0.500 to 0.875 once " +
      "I added patterns for \"shall not exceed\" and \"consequential damages\". GoverningLaw sat " +
      "stubbornly at 0.40 in the keyword-only baseline because it kept getting confused with Jurisdiction; " +
      "the ML model resolved that on its own."
  ),

  h2("5.2 Retriever Performance"),
  p(
    "I report retriever quality with Hit@5, which is the fraction of evaluation clauses for which the " +
      "ground-truth ICA section ends up somewhere in the top five results. The table below covers all 171 " +
      "evaluation clauses across the ten clause types I have ground-truth labels for."
  ),
  buildTable(
    [
      ["Clause Type", "BM25 only — Hit@5", "Statute map + BM25 — Hit@5", "n"],
      ["Arbitration", "0.97", "1.00", "33"],
      ["Termination", "0.00", "1.00", "17"],
      ["PaymentTerms", "0.06", "1.00", "16"],
      ["IPAssignment", "0.07", "1.00", "30"],
      ["NonCompete", "0.71", "1.00", "17"],
      ["Confidentiality", "0.25", "1.00", "12"],
      ["ForceMajeure", "0.00", "1.00", "14"],
      ["Indemnification", "0.25", "1.00", "4"],
      ["Renewal", "0.00", "1.00", "19"],
      ["LiabilityCap", "0.00", "1.00", "9"],
      ["Overall", "0.23", "1.00", "171"],
    ],
    { widths: [25, 25, 30, 20] }
  ),
  spacer(),
  p(
    "The contrast in the last column is the central finding of the retriever experiments. BM25 alone " +
      "lands at 23.0% Hit@5 overall, while the same BM25 layer with the statute map switched on lands at " +
      "100%. The cause is structural rather than statistical: Indian statute is written in compressed " +
      "Victorian English (\"every agreement by which any one is restrained from exercising a lawful " +
      "profession...\") while a typical contract clause uses the opposite vocabulary (\"the Service " +
      "Provider shall not engage in any competing business...\"). Lexical search misses the link because " +
      "the words are not the same; semantic search misses it because the InLegalBERT representation of " +
      "modern commercial drafting and 1872 statutory drafting are not as close together as one might " +
      "hope. A small static map closes the gap for the clause types where I know the answer in advance, " +
      "and BM25 (or FAISS) handles the rest. The general lesson, which I think applies beyond this " +
      "project, is that when the target set is small and known, a hand-built lookup is worth more than " +
      "any amount of clever similarity learning."
  ),

  h2("5.3 End-to-End Runtime and Footprint"),
  p(
    "End-to-end timing on the i7-1260P laptop comes out at roughly 4.1 seconds for the 15-clause MSA on " +
      "a cold start and around 1.6 seconds once caches are warm. The bulk of the cold-start cost is " +
      "loading and running InLegalBERT for the FAISS layer; once the model is in memory, the embedding " +
      "of a single clause is cheap and is reused by the classifier's Stage 3 fallback. The disk footprint " +
      "is about 750 MB end to end, dominated by the InLegalBERT weights. If the user is willing to give " +
      "up FAISS and the embedding fallback, the system runs in roughly 60 MB on statute map plus BM25. " +
      "At no point during analysis does any part of the pipeline reach out over the network."
  ),

  h2("5.4 Error Analysis"),
  p(
    "The misclassifications are not random. Three patterns account for most of what the classifier gets " +
      "wrong, and each of them points to a specific next step rather than to a generic \"more data\" wish."
  ),
  bullet(
    "Termination versus Renewal. Indian commercial drafting often packs the initial term, the renewal mechanism and the termination-for-convenience right into a single paragraph. The classifier sees more termination-related keywords than renewal-related ones in that paragraph, so it tends to label the whole thing as Termination and miss the renewal signal entirely. A pre-processor that splits long compound paragraphs at semantic boundaries would probably fix most of these cases."
  ),
  bullet(
    "Jurisdiction versus GoverningLaw. These two clauses sit next to each other in almost every contract and Indian drafters routinely merge them into one sentence. The classifier mis-allocates about one in five examples between them. Either a multi-label head (so a single clause can be both) or a sentence-level splitter before classification would help."
  ),
  bullet(
    "Statute-citation noise. Some contracts quote a section of the Arbitration and Conciliation Act, 1996 inside the body of a clause that is not really about arbitration. The Stage 2 Arbitration rule has a blocklist for the most common pattern (\"section N of the arbitration\"), but a cleaner approach would be to detect statute-citation spans up front and exclude them from rule matching altogether."
  ),
  p(
    "Two limitations deserve a direct call-out because they affect how the headline numbers should be " +
      "read. First, CUAD is built from US commercial contracts. Clause-type detection transfers across " +
      "jurisdictions reasonably well because the language patterns are similar, but rare Indian-specific " +
      "clauses — stamp duty allocation, GST gross-up, DPDP-style data-processing addenda — are " +
      "under-represented in training and currently lean almost entirely on the keyword rules. Second, I " +
      "did not finish the Label Studio annotation. 212 tasks are prepared and ready to import, but they " +
      "were not annotated by a human lawyer, so the per-type F1 numbers above are computed against " +
      "scraped labels rather than expert-verified gold. The numbers are useful as a relative comparison " +
      "between system versions, but they should not be over-read as absolute accuracy."
  ),

  h2("5.5 Failed Approaches"),
  p(
    "Three things I tried did not work, and I think they are worth recording for anyone who plans to do " +
      "something similar. The first was trying to scrape more Indemnification and LiabilityCap clauses " +
      "from the Indian Kanoon judgement JSONs to balance out CUAD. Indian courts discuss these clauses in " +
      "their own prose instead of quoting the clause text verbatim, so my extraction patterns came back " +
      "almost empty. I ended up accepting that coverage and leaning on the keyword rules for those types. " +
      "The second was trying to render the report inside the browser preview using document.write(); " +
      "this broke the evaluation context's 30-second timeout because the document.write call counted as a " +
      "page navigation. I added a dedicated /report Flask route to sidestep the problem. The third was " +
      "using Flask's send_from_directory to serve the generated report file, which returned 404 errors " +
      "on Flask 3.x even with the path apparently correct. Switching to send_file with an explicit " +
      "pathlib.Path object fixed it. None of these were major setbacks, but each one cost a few hours " +
      "and was worth documenting."
  ),
  pageBreak(),
];

// 5. CONCLUSION
const conclusion = [
  h1("6. Conclusion"),
  p(
    "The thing I set out to test was whether you could build a useful Indian-law contract reviewer that " +
      "runs entirely on a laptop, never phones home, and explains itself by pointing at the statute " +
      "rather than by sounding fluent. Six months later, the answer is yes, with a couple of asterisks. " +
      "The single most useful piece of evidence from the experiments is that a hand-curated dictionary of " +
      "fewer than thirty entries — the statute map — moves retrieval Hit@5 from 23.0% to 100% on the " +
      "evaluation set. That is a much bigger gain than any of the dense-retrieval improvements I tried, " +
      "and it changes how I think about where to put effort in this kind of system. The classifier, at " +
      "83.2% overall accuracy, is good enough to drive a triage tool for non-lawyers, and the three-stage " +
      "cascade means that when it does go wrong, the path it took is auditable end to end."
  ),
  p(
    "What this project is not is also worth stating clearly. It is not legal advice and it is not a " +
      "lawyer; the generated report carries a disclaimer in the UI, in the HTML and in the JSON export. " +
      "The decision to skip an LLM in the generator was deliberate. It costs me the fluency that a " +
      "modern model would add to the explanations, but it gives me three things in return: the report " +
      "never invents a section number, it runs offline, and no contract text ever leaves the user's " +
      "machine. For a tool that someone might rely on before signing something binding, I would rather " +
      "err on the side of being too conservative than too confident."
  ),
  p(
    "There is also a software-engineering point worth making. A lot of what makes this pipeline " +
      "deployable today is not the classifier or the retriever but the scaffolding around them — pinned " +
      "dependency versions, a 116-test pytest suite that runs in CI, graceful degradation when FAISS or " +
      "OCR is missing, explicit /health and /demo routes, and a Dockerfile. The classifier could be " +
      "swapped for a fine-tuned InLegalBERT tomorrow without disturbing any of that. The boring " +
      "scaffolding is what made the model improvements safe to try."
  ),
  pageBreak(),
];

// 6. FUTURE WORK
const futureWork = [
  h1("7. Future Work"),
  p(
    "Several things would be worth doing next. I have ordered them by roughly how much value I think " +
      "each would add given the effort required."
  ),
  bullet(
    "Finish the Label Studio annotation. 212 tasks are already prepared and waiting; the missing piece is a qualified Indian advocate willing to label them. Once that is done, the classifier numbers can be re-computed against a properly verified gold set and reported as absolute rather than indicative accuracy."
  ),
  bullet(
    "Swap the TF-IDF and Logistic Regression model for a fine-tuned InLegalBERT classification head. A quick prototype I ran suggests a four to six point F1 gain is realistic, at the cost of slower inference and a much larger model artefact."
  ),
  bullet(
    "Make the classifier multi-label. A lot of Indian clauses combine Termination and Renewal, or GoverningLaw and Jurisdiction, in a single paragraph, and the current single-label output forces a winner where there should not be one."
  ),
  bullet(
    "Extend the knowledge base beyond the ICA. The obvious additions are the Arbitration and Conciliation Act, 1996, the Specific Relief Act, 1963, the Sale of Goods Act, 1930, Sections 43A and 79 of the Information Technology Act, 2000, and the Digital Personal Data Protection Act, 2023."
  ),
  bullet(
    "Add case-law retrieval as a second knowledge-base layer, so that a non-compete clause can be flagged with both Section 27 and the relevant Supreme Court judgements (Niranjan Shankar Golikari, Krishan Murgai, Percept D'Mark). The infrastructure for this is essentially the same as for the statute retriever."
  ),
  bullet(
    "Add an optional, fully-local LLM generation layer using Llama 3 8B Instruct or Mistral 7B Instruct via Ollama. It would write a more natural paragraph-level summary, while citations would still come from the deterministic retriever. The template version should stay as the default for users who care about reproducibility."
  ),
  bullet(
    "Add Hindi support. The UI strings, the explanation templates and a clause-language-detection step in the segmenter are all in scope; the harder question is what to do with mixed Hindi-English contracts, which are common in MSME procurement."
  ),
  bullet(
    "Add a redlining mode that suggests specific edits to high-risk clauses. The obvious starting point is non-compete: the system already knows Section 27 is the relevant statute, so it can suggest narrowing the geographic scope or limiting the restriction to the duration of employment."
  ),
  bullet(
    "Run a user study with a legal-aid clinic. The right thing to measure is whether non-lawyer users make materially better signing decisions when they have the tool, compared with reading the contract on their own. I suspect they do, but the only way to find out is to measure."
  ),
  pageBreak(),
];

// 7. REFERENCES
const references = [
  h1("8. References"),
  refItem(
    "[1] Hendrycks, D., Burns, C., Chen, A. and Ball, S. (2021). CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review. In Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track."
  ),
  refItem(
    "[2] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S. and Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems 33 (NeurIPS 2020)."
  ),
  refItem(
    "[3] Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N. and Androutsopoulos, I. (2020). LEGAL-BERT: The Muppets Straight Out of Law School. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 2898–2904."
  ),
  refItem(
    "[4] Paul, S., Goyal, P. and Ghosh, S. (2022). Pre-training Transformers on Indian Legal Text. arXiv preprint arXiv:2209.06049. (Release of InLegalBERT by IIT Kharagpur.)"
  ),
  refItem(
    "[5] Robertson, S. and Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4), pp. 333–389."
  ),
  refItem(
    "[6] Johnson, J., Douze, M. and Jégou, H. (2017). Billion-scale Similarity Search with GPUs. arXiv preprint arXiv:1702.08734. (FAISS.)"
  ),
  refItem(
    "[7] Cormack, G. V., Clarke, C. L. A. and Buettcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. In Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 758–759."
  ),
  refItem(
    "[8] Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D. and Yih, W. (2020). Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6769–6781."
  ),
  refItem(
    "[9] Lin, J., Ma, X., Lin, S.-C., Yang, J.-H., Pradeep, R. and Nogueira, R. (2021). Pyserini: A Python Toolkit for Reproducible Information Retrieval Research with Sparse and Dense Representations. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 2356–2362."
  ),
  refItem(
    "[10] Bhattacharya, P., Hiware, K., Rajgaria, S., Pochhi, N., Ghosh, K. and Ghosh, S. (2019). A Comparative Study of Summarization Algorithms Applied to Legal Case Judgments. In Advances in Information Retrieval (ECIR 2019), Lecture Notes in Computer Science, vol. 11437, pp. 413–428."
  ),
  refItem(
    "[11] Kalamkar, P., Agarwal, A., Tiwari, A., Gupta, S., Karn, S. and Raghavan, V. (2022). Named Entity Recognition in Indian Court Judgments. arXiv preprint arXiv:2211.03442. (OpenNyAI corpus.)"
  ),
  refItem(
    "[12] Surden, H. (2014). Machine Learning and Law. Washington Law Review, 89(1), pp. 87–115."
  ),
  refItem(
    "[13] Surden, H. (2019). Artificial Intelligence and Law: An Overview. Georgia State University Law Review, 35(4), pp. 1305–1337."
  ),
  refItem(
    "[14] Bench-Capon, T. (2020). The Need for Good Old-Fashioned AI and Law. In Proceedings of the 2020 International Conference on Legal Knowledge and Information Systems (JURIX), pp. 145–154."
  ),
  refItem(
    "[15] Hachey, B. and Grover, C. (2006). Extractive Summarisation of Legal Texts. Artificial Intelligence and Law, 14(4), pp. 305–345."
  ),
  refItem(
    "[16] Moens, M.-F. and Boiy, E. (2007). Detection of Arguments in Legal Texts. In Proceedings of the 11th International Conference on Artificial Intelligence and Law (ICAIL), pp. 225–230."
  ),
  refItem(
    "[17] Government of India. (1872). The Indian Contract Act, 1872. Act No. 9 of 1872 (as amended). Primary statutory source for the knowledge base used in this project."
  ),
  refItem(
    "[18] Government of India. (1996). The Arbitration and Conciliation Act, 1996. Act No. 26 of 1996 (as amended)."
  ),
  refItem(
    "[19] Government of India. (2023). The Digital Personal Data Protection Act, 2023. Act No. 22 of 2023."
  ),
  refItem(
    "[20] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, pp. 2825–2830."
  ),
  refItem(
    "[21] The Atticus Project. (2021). CUAD Dataset and Annotation Guidelines. https://www.atticusprojectai.org/cuad (accessed May 2026)."
  ),
  refItem(
    "[22] Indian Kanoon. Indian case-law database used as the source of evaluation judgements. https://indiankanoon.org (accessed May 2026)."
  ),
];

// ---------- assemble document ----------

const sections = [
  {
    properties: {
      page: {
        margin: {
          top: convertInchesToTwip(1),
          right: convertInchesToTwip(1),
          bottom: convertInchesToTwip(1),
          left: convertInchesToTwip(1),
        },
      },
    },
    footers: {
      default: new Footer({
        children: [
          new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [
              new TextRun({
                children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES],
                size: 20,
                font: FONT,
              }),
            ],
          }),
        ],
      }),
    },
    headers: {
      default: new Header({
        children: [
          new Paragraph({
            alignment: AlignmentType.RIGHT,
            children: [
              new TextRun({
                text: "RAG-Based Legal Contract Analyzer — Project Report",
                italics: true,
                size: 18,
                font: FONT,
                color: "595959",
              }),
            ],
          }),
        ],
      }),
    },
    children: [
      ...titlePage,
      ...abstract,
      ...introduction,
      ...literature,
      ...methodology,
      ...experiments,
      ...results,
      ...conclusion,
      ...futureWork,
      ...references,
    ],
  },
];

const doc = new Document({
  creator: "Aniketh Vustepalle",
  title: "RAG-Based Legal Contract Analyzer — Project Report",
  description:
    "Academic project report on an offline RAG pipeline for Indian commercial contract review",
  styles: {
    default: {
      document: {
        run: { font: FONT, size: 22 },
        paragraph: { spacing: { line: 300 } },
      },
    },
  },
  sections,
});

Packer.toBuffer(doc).then((buf) => {
  const outPath = path.join(
    "C:",
    "Users",
    "Ani",
    "OneDrive",
    "Desktop",
    "legal contract Analyzer",
    "Legal_Contract_Analyzer_Report.docx"
  );
  fs.writeFileSync(outPath, buf);
  console.log("Wrote", outPath, "size:", buf.length, "bytes");
});
