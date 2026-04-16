import os
import re
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import TypedDict, List

from langchain_groq import ChatGroq
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from sentence_transformers import SentenceTransformer
import chromadb


class CapstoneState(TypedDict):
    question: str
    messages: list
    route: str
    retrieved: str
    sources: list
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    user_name: str
    document_type: str


DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Indian Contract Act 1872 — Void vs Voidable Contracts",
        "text": """
Under the Indian Contract Act 1872, a void contract and a voidable contract are 
fundamentally different in legal consequence. A void contract under Section 2(g) 
is one that is not enforceable by law from the very beginning — it is ab initio void 
and creates no legal rights or obligations between the parties. Examples include 
agreements made by a minor (Section 11), agreements without consideration (Section 25), 
agreements in restraint of trade (Section 27), and agreements to do an impossible act 
(Section 56). A void agreement cannot be ratified or validated by the parties.

A voidable contract under Section 2(i) is one that is enforceable at the option of one 
or more of the parties but not at the option of the other. Under Section 19, contracts 
induced by coercion, fraud, or misrepresentation are voidable at the option of the 
aggrieved party. Under Section 19A, contracts induced by undue influence are also 
voidable. The key distinction is that a voidable contract remains valid until the 
aggrieved party exercises the right to rescind it. Rescission must be communicated 
within a reasonable time and before third-party rights have been acquired.

Practical example: A contract signed under threat of violence is voidable (Section 15 
coercion — Section 19). A contract for sale of smuggled goods is void as it is 
opposed to public policy under Section 23. A contract with a person of unsound mind 
is void under Section 11. Once a voidable contract is rescinded, the parties must 
restore benefits received under Section 64. Courts have held that the burden of 
proving coercion or undue influence lies on the party asserting it.
        """
    },
    {
        "id": "doc_002",
        "topic": "Doctrine of Frustration — Section 56 Indian Contract Act",
        "text": """
Section 56 of the Indian Contract Act 1872 codifies the doctrine of frustration. 
Paragraph 1 states that an agreement to do an impossible act is void. Paragraph 2 
states that a contract to do an act which after the contract is made becomes 
impossible or unlawful by reason of some event which the promisor could not prevent 
becomes void when the act becomes impossible or unlawful.

The doctrine was elaborated in Taylor v Caldwell (1863) by the English courts and 
adopted in Indian law. The landmark Indian case is Satyabrata Ghose v Mugneeram 
Bangur (1954 AIR SC 44), where the Supreme Court held that Section 56 must be 
interpreted broadly to include not just literal physical impossibility but also 
commercial frustration — where the performance becomes radically different from 
what was undertaken. The court rejected the narrow English approach and adopted 
a wider Indian interpretation.

Frustration does NOT apply in the following situations: where the supervening event 
was foreseeable and the party assumed the risk; where the contract itself allocates 
the risk to one party (force majeure clauses); where the frustration is self-induced 
by one party's conduct; or where performance merely becomes more expensive or 
difficult. The effect of frustration is automatic discharge of both parties — no 
court order is needed. However, under Section 65, parties must restore any benefit 
received before the contract became void. COVID-19 lockdown cases in India in 2020-21 
saw courts examine frustration arguments carefully, generally rejecting them for 
contracts where risk allocation was clear or alternatives existed.
        """
    },
    {
        "id": "doc_003",
        "topic": "NDA Clauses — Enforceability and Carve-outs in India",
        "text": """
Non-Disclosure Agreements (NDAs) in India are governed by the Indian Contract Act 1872 
and the Information Technology Act 2000. A well-drafted NDA must clearly define: 
(1) what constitutes confidential information — specific definition prevents disputes; 
(2) the obligations of the receiving party — non-disclosure, non-use, restricted access; 
(3) the duration of confidentiality obligation — courts prefer specific time limits; 
(4) permitted disclosures and carve-outs; and (5) remedies for breach.

Standard carve-outs (exceptions to confidentiality) include: information already in 
the public domain before disclosure; information independently developed by the 
receiving party without use of disclosed information; information received from a 
third party without restriction; information required to be disclosed by law, court 
order, or regulatory authority (with notice to disclosing party where possible); 
and information already known to the receiving party before disclosure (provable 
by prior written records). The residuals clause — allowing use of information 
retained in unaided human memory — is common in technology NDAs but contested 
in India as it creates a significant exception.

Enforceability concerns: An NDA covering information that is not actually confidential, 
or one with unlimited duration, may face challenges. Courts have held that overly broad 
NDAs may partially fail for uncertainty. Remedies available for breach include: 
injunctive relief under Section 38 of the Specific Relief Act (urgent cases); damages 
for actual loss proven; account of profits if misuse led to unjust enrichment; and 
criminal action under Section 72 of the IT Act for disclosure of information accessed 
in breach of lawful contract. Jurisdiction clause must specify Indian courts or 
arbitration clearly.
        """
    },
    {
        "id": "doc_004",
        "topic": "Employment Contracts — Restraint of Trade and Non-Compete Clauses",
        "text": """
Section 27 of the Indian Contract Act 1872 declares that every agreement in restraint 
of trade is void. This provision is interpreted far more strictly in India than in 
England or the United States. Indian courts have consistently held that post-termination 
non-compete clauses — clauses that prevent an employee from working for a competitor 
after leaving employment — are void under Section 27 and unenforceable.

Key cases: Niranjan Shankar Golikari v Century Spinning (1967 AIR SC 1098) is the 
leading case, which held that restraints operative DURING employment are valid but 
post-employment restraints are void. FL Smidth v FLSmidth Rahul (2011) Bombay HC 
reaffirmed that even for senior employees with access to trade secrets, post-employment 
non-compete is void. However, the courts do enforce: (1) non-solicitation of clients 
during employment; (2) confidentiality obligations (these are not restraint of trade); 
(3) garden leave provisions during the notice period — the employee is paid full salary 
but required to stay away from work; (4) IP assignment clauses requiring invention 
disclosure and assignment during employment.

Practical implications: A paralegal reviewing an employment contract should flag any 
post-termination non-compete as legally unenforceable in India. The employer's only 
real protection post-employment is through robust confidentiality and NDA clauses, 
trade secret protection, and IP assignment. Non-solicitation of employees (poaching 
restrictions) is in a grey area — some courts have upheld reasonable restrictions. 
Notice period clauses and buy-out provisions are enforceable provided they are 
reasonable and not a penalty.
        """
    },
    {
        "id": "doc_005",
        "topic": "Civil Procedure Code — Order VII Rule 11, Res Judicata, Res Sub Judice",
        "text": """
Order VII Rule 11 of the Civil Procedure Code (CPC) 1908 provides grounds for rejection 
of a plaint at the threshold stage, without calling the defendant. Grounds for rejection 
include: (a) the plaint does not disclose a cause of action; (b) the relief claimed is 
undervalued and the plaintiff does not correct the valuation when directed; (c) the 
plaint is on insufficient stamp paper and not rectified; (d) the suit appears from the 
statement in the plaint to be barred by any law (including limitation). The rejection 
operates as a decree and is appealable. Importantly, rejection of plaint does not 
preclude refiling a fresh suit if the defect is curable.

Res Sub Judice under Section 10 CPC: Where a suit involving the same parties and same 
subject matter is already pending in a competent court, the subsequently filed court 
must stay proceedings. This prevents multiplicity of proceedings on the same issue. 
Section 10 applies only if both courts are competent to grant the same relief and the 
matter in issue is directly and substantially the same.

Res Judicata under Section 11 CPC: Once a matter has been finally decided by a 
competent court, the same parties cannot re-litigate the same matter. The doctrine 
bars a second suit if: (1) the matter was directly and substantially in issue in the 
former suit; (2) the former suit was between the same parties or parties under whom 
they claim; (3) the parties litigated under the same title; and (4) the court that 
decided the former suit was competent to try the subsequent suit. Constructive res 
judicata (Explanation IV to Section 11) bars issues that ought to have been raised 
in the former suit.
        """
    },
    {
        "id": "doc_006",
        "topic": "Limitation Act 1963 — Articles, Computation, Acknowledgement, Condonation",
        "text": """
The Limitation Act 1963 prescribes time limits for filing suits, appeals, and 
applications. Key articles relevant to civil litigation: Article 36 — recovery of 
money lent under a written contract: 3 years from date fixed for repayment. Article 55 
— compensation for breach of contract: 3 years from date when contract is broken. 
Article 113 — residuary article for suits not covered elsewhere: 3 years from when 
right to sue accrues. Article 58 — declaration: 3 years from when right to sue 
first accrues.

Computation of limitation under Section 12: Time for obtaining a copy of the decree 
or order appealed from is excluded. Section 14 excludes time spent prosecuting in 
good faith before a court that lacked jurisdiction, provided there was no fault of 
the plaintiff.

Section 18 — Acknowledgement: If the person against whom a right is claimed 
acknowledges the right in writing signed by them (or their authorised agent) before 
expiry of the limitation period, a fresh period of limitation runs from the date of 
acknowledgement. The acknowledgement need not amount to a promise to pay — mere 
acknowledgement of liability is sufficient. The writing must exist before the original 
limitation period expires.

Section 5 — Condonation of Delay: Appeals and applications (not suits) can be filed 
after the limitation period if the applicant satisfies the court that there was 
sufficient cause for the delay. Courts apply this liberally for genuine cases. 
Sufficient cause has been held to include illness, mistake of counsel, and official 
correspondence. However, suits cannot benefit from Section 5 — only appeals 
and applications.
        """
    },
    {
        "id": "doc_007",
        "topic": "Legal Notice under Section 80 CPC — Drafting, Service, and Validity",
        "text": """
Section 80 of the Civil Procedure Code 1908 mandates that before filing a suit against 
the Government of India, a State Government, or a public officer acting in official 
capacity, a notice of 2 months must be given. The notice must state: the name, 
description and place of residence of the plaintiff; the cause of action; and the 
relief claimed. Failure to give notice is a procedural bar — the suit is not dismissed 
but stayed until notice requirements are met. The proviso to Section 80 allows urgent 
interim relief without prior notice, subject to the court's permission.

For suits against private parties, a legal notice is not statutorily mandated under 
CPC but is common practice and commercially important. It creates a paper trail, 
demonstrates good faith, and often prompts settlement without litigation.

Service of legal notice: Notices should be sent by registered post with acknowledgment 
due (RPAD) to the registered address or last known address. Service by courier with 
tracking is also acceptable. WhatsApp service: The Bombay High Court in Tata Sons 
v John Doe (2017) and subsequent cases have held that service via WhatsApp can be 
valid where delivery and read receipts are shown (double blue ticks), but this is 
supplementary and not the primary mode for formal legal notices. Electronic mail 
notice is valid if the contract specifies email as the address for notice.

A defendant who receives a legal notice and does not respond within the stipulated 
period (typically 15 to 30 days for private parties) cannot later claim the dispute 
was not raised before suit. Non-response strengthens the plaintiff's case for costs.
        """
    },
    {
        "id": "doc_008",
        "topic": "Bail Jurisprudence — Sections 436, 437, 438, 439 CrPC",
        "text": """
Under the Code of Criminal Procedure 1973, bail provisions operate differently for 
bailable and non-bailable offences. Section 436 CrPC: For bailable offences, bail 
is a matter of right. The accused can demand bail from the police officer or the 
court and it must be granted. The officer or court has no discretion to refuse.

Section 437 CrPC: For non-bailable offences, bail is at the discretion of the 
Magistrate. The Magistrate must refuse bail if there appear reasonable grounds to 
believe the accused is guilty of an offence punishable with death or life imprisonment. 
For other non-bailable offences, the triple test applies in practice: (1) flight risk 
— likelihood of the accused absconding; (2) tampering with evidence or witnesses; 
(3) risk of repeat offence. The Supreme Court in Arnesh Kumar v State of Bihar (2014) 
issued guidelines requiring police to justify arrests in offences punishable up to 
7 years and requiring Magistrates to apply mind before remanding to custody.

Section 438 CrPC — Anticipatory Bail: A person who has reason to believe they may 
be arrested for a non-bailable offence can apply to the Sessions Court or High Court 
for a direction that they be released on bail upon arrest. This is a pre-arrest remedy. 
The court considers: the nature and gravity of accusation; the applicant's antecedents; 
the possibility of flight; and whether the accusation is made to humiliate or injure. 
Section 439 CrPC: Sessions Court and High Court have special powers to grant bail in 
cases of non-bailable offences and can impose conditions such as surrendering passport, 
marking attendance at police station, and restricting travel.
        """
    },
    {
        "id": "doc_009",
        "topic": "Power of Attorney — GPA vs SPA and Supreme Court Ruling on Property",
        "text": """
A Power of Attorney (PoA) is a legal instrument under the Powers of Attorney Act 1882 
authorising an agent (attorney) to act on behalf of the principal in specified matters. 
A General Power of Attorney (GPA) grants broad authority across multiple transactions 
or matters. A Specific Power of Attorney (SPA) is limited to a particular act or 
transaction, such as registering a specific property document.

The landmark Supreme Court ruling in Suraj Lamp and Industries Pvt Ltd v State of 
Haryana (2011) 11 SCC 438 fundamentally changed GPA-based property transactions. 
The court held that GPA/SA (sale agreement) transactions cannot be treated as valid 
modes of transfer of immovable property. Property can only be legally transferred by 
a registered sale deed. GPA-based property sales, even if notarized, do not confer 
title. This ruling applied prospectively — transactions before 2011 were not affected. 
Post this ruling, GPA is valid for authorising an agent to execute a registered sale 
deed on the principal's behalf, but the underlying sale deed must still be registered.

Registration requirements: A PoA executed in India for immovable property transactions 
must be registered under the Registration Act 1908 (Section 17) to be valid. A PoA 
executed abroad must be notarized before an Indian Consulate or notary public and 
adjudicated for stamp duty within 3 months of arrival in India.

Revocation: A PoA is automatically revoked on the death, insanity, or insolvency of 
the principal. It can be expressly revoked by notice to the attorney and to any third 
parties who have relied on it. An irrevocable PoA (coupled with interest) — where 
the attorney has a personal interest in the authority — cannot be revoked without 
the attorney's consent.
        """
    },
    {
        "id": "doc_010",
        "topic": "Confidentiality Clauses — Injunctive Relief and Damages",
        "text": """
Breach of a confidentiality clause gives rise to both civil and equitable remedies. 
In India, the primary remedies are injunctive relief under the Specific Relief Act 
and damages under the Indian Contract Act. Section 38 of the Specific Relief Act 1963 
(as amended in 2018) provides for perpetual injunctions to prevent breach of an 
obligation arising from contract. Courts grant injunctions when: (1) there is a strong 
prima facie case of breach; (2) the balance of convenience favours the applicant — 
i.e., the harm from refusing the injunction outweighs the harm of granting it; and 
(3) the applicant would suffer irreparable harm not compensable in money.

The irreparable harm standard is critical — courts recognise that confidential 
information, once disclosed, cannot be "undisclosed." This makes injunctive relief 
the natural remedy for threatened or ongoing breach. A temporary injunction under 
Order XXXIX Rule 1 and 2 CPC can be obtained at an ex parte hearing in urgent cases.

Calculating damages for breach of confidentiality: Courts award (1) actual loss 
suffered — loss of business, clients, or contracts provably caused by the breach; 
(2) account of profits — the profit made by the defendant through misuse of 
confidential information; (3) nominal damages where breach is proven but loss 
is not quantifiable. Liquidated damages clauses specifying a sum for breach are 
enforceable under Section 74 of the Indian Contract Act provided they represent 
a genuine pre-estimate of loss and are not a penalty. Courts can reduce a penalty 
clause to reasonable compensation under Section 74.
        """
    },
    {
        "id": "doc_011",
        "topic": "Evidence Act — Admissibility of Digital Evidence and Section 65B",
        "text": """
Section 65B of the Indian Evidence Act 1872 governs the admissibility of electronic 
records as evidence. Electronic records — including emails, WhatsApp messages, 
digital contracts, CCTV footage, computer outputs, and electronic documents — are 
admissible as evidence only if accompanied by a Section 65B certificate. The 
certificate must be issued by a responsible official of the computer or system 
from which the electronic record was produced, certifying: (1) the electronic 
record was produced by a computer during ordinary use; (2) the computer was 
operating properly; (3) the information in the record reproduces information 
that was supplied to the computer in the ordinary course of activities.

The Supreme Court in Arjun Panditrao Khotkar v Kailash Kushanrao Gorantyal (2020) 
7 SCC 1 settled the law definitively: a Section 65B certificate is mandatory and 
cannot be waived. Secondary evidence of electronic records cannot be led without 
the certificate. The certificate must be obtained from whoever owns/operates the 
relevant computer system.

Practical implications for legal document work: WhatsApp messages produced in 
litigation must be accompanied by a Section 65B certificate from the mobile 
device owner or the phone company. Screenshots alone are insufficient. Emails 
require a certificate from the email server administrator or the organisation's 
IT head. Electronic contracts — those signed using digital signatures under the 
IT Act 2000 — are valid contracts under Section 10A of the IT Act and are 
admissible without Section 65B if they carry a valid digital signature.
        """
    },
    {
        "id": "doc_012",
        "topic": "Arbitration and Conciliation Act 1996 — Section 8, 34, 37 and Seat vs Venue",
        "text": """
The Arbitration and Conciliation Act 1996 is the primary legislation governing 
arbitration in India, based on the UNCITRAL Model Law. Section 8: If a party to 
an arbitration agreement files a civil suit on a matter covered by the agreement, 
the other party can apply to the court to refer the parties to arbitration. The 
court must refer the parties to arbitration unless it finds that the arbitration 
agreement is null, void, inoperative, or incapable of performance. The application 
must be filed before filing the first statement on the merits (the written statement).

Section 34 — Setting Aside an Arbitral Award: An arbitral award can be challenged 
in court within 3 months of receiving the award (extendable by 30 days for sufficient 
cause under Section 34(3)). Grounds for setting aside: (a) incapacity of a party or 
invalidity of agreement; (b) no proper notice or inability to present case; (c) award 
deals with matters beyond scope of submission; (d) composition of tribunal or procedure 
not in accordance with agreement; (e) subject matter not arbitrable under Indian law; 
(f) award conflicts with public policy of India (including fraud or corruption in 
the award or patent illegality on the face of the award — domestic arbitrations only).

Section 37: Appeals lie against specific orders — refusal to refer to arbitration 
(Section 8), granting or refusing an interim measure (Section 9), setting aside or 
refusing to set aside an award (Section 34).

Seat vs Venue: The seat of arbitration determines the juridical home — which country's 
courts have supervisory jurisdiction and which procedural law applies. Venue is merely 
the physical location of hearings. In BALCO v Kaiser Aluminium (2012) 9 SCC 552, 
the Supreme Court held that the seat confers exclusive jurisdiction. Clauses saying 
"venue: Mumbai" without specifying seat create disputes — courts generally treat a 
fixed exclusive venue as the seat, but the distinction must be explicit in drafting.
        """
    },
    {
        "id": "doc_013",
        "topic": "Consumer Protection Act 2019 — Jurisdiction, Deficiency, Product Liability",
        "text": """
The Consumer Protection Act 2019 replaced the 1986 Act with enhanced provisions. 
Pecuniary jurisdiction: District Consumer Commission — claims up to Rs. 1 crore; 
State Consumer Commission — claims between Rs. 1 crore and Rs. 10 crore; National 
Consumer Commission — claims above Rs. 10 crore. Territorial jurisdiction: complaint 
can be filed where the complainant resides or works (a significant shift from the 1986 
Act that required filing where the opposite party resided or had its principal office).

Section 2(7) defines 'consumer': a person who buys goods or avails services for 
consideration, and includes beneficiaries. A person who buys goods for commercial 
purpose (resale or profit) is excluded. However, self-employed persons using goods 
for earning their livelihood are included.

Deficiency of service under Section 2(11): any fault, imperfection, shortcoming, or 
inadequacy in quality, nature, or manner of performance required under law or contract. 
This is broad and covers banking, insurance, medical, legal, and e-commerce services.

Product liability under Chapter VI: A manufacturer, product service provider, or 
product seller can be held liable for a defective product or deficient service causing 
harm. Liability arises if the product had a manufacturing defect, was not accompanied 
by adequate instructions for correct use, or did not conform to express warranty. 
E-commerce liability: platforms are required to display return policies, seller 
information, and are liable for unfair trade practices on their platform. Misleading 
advertisements by celebrities now attract liability under Section 21.
        """
    },
    {
        "id": "doc_014",
        "topic": "IPC Sections for Corporate and Financial Fraud",
        "text": """
Corporate fraud in India attracts both criminal provisions under the Indian Penal 
Code 1860 and specific provisions under the Companies Act 2013. Key IPC sections:

Section 415/420 IPC — Cheating: Dishonestly inducing a person to deliver property 
or to do or omit to do anything which they would not do if not deceived, causing 
damage. Section 420 (cheating and dishonestly inducing delivery of property) is 
punishable with up to 7 years imprisonment. Essential ingredients: deception by 
the accused; deception was dishonest; person deceived acted on the deception; 
and this caused damage to the deceived party.

Section 405/406 IPC — Criminal Breach of Trust: A person entrusted with property 
who dishonestly misappropriates or uses it in violation of the trust commits criminal 
breach of trust. Section 406 carries up to 3 years. Section 409 (CBT by a public 
servant, banker, merchant, or agent) carries up to 10 years. Directors managing 
company funds are treated as trustees in several High Court judgments.

Section 463/467/468/471 IPC — Forgery: Section 467 (forgery of valuable security, 
will, authority to make valuable security) carries life imprisonment. Section 468 
(forgery for purpose of cheating) carries 7 years. Section 471 (using as genuine a 
document known to be forged) carries the same punishment as for making the forged 
document. These are commonly invoked in cheque fraud, land record manipulation, 
and corporate document fraud.

Companies Act 2013 Section 447 — Fraud: Any person who is found guilty of fraud 
(involving assets of Rs. 10 lakh or more or 1% of turnover, whichever is lower) 
faces imprisonment of 6 months to 10 years and fine up to 3 times the fraud amount. 
Section 447 applies to directors, auditors, company secretaries, and advisors. 
The Serious Fraud Investigation Office (SFIO) investigates serious corporate fraud.
        """
    },
    {
        "id": "doc_015",
        "topic": "Specific Relief Act 2018 Amendment — Mandatory Specific Performance",
        "text": """
The Specific Relief (Amendment) Act 2018 fundamentally changed the discretionary 
nature of specific performance in India. Prior to the amendment, Section 20 gave 
courts discretion to grant or refuse specific performance even where a breach was 
proven — courts frequently refused and awarded damages instead. The 2018 amendment 
substituted Section 20 to make specific performance a right, not a discretion.

Post-amendment, courts must grant specific performance of a contract unless: (a) the 
contract involves performance of a continuous duty which the court cannot supervise; 
(b) the contract is so dependent on personal qualifications of the party that the 
court cannot enforce it; (c) the contract is in its nature determinable. The amendment 
was particularly aimed at infrastructure and construction contracts where delays caused 
by contractors could not be adequately compensated through damages.

Section 20A (newly inserted by 2018 amendment) — Substituted Performance: Where 
a contract is broken, the party who suffers the breach may get the contract performed 
by a third party or by his own agency and recover expenses and costs from the party 
in breach. This must be done after providing a notice to the party in breach. The 
notice must give a reasonable time to perform. If the party in breach performs within 
that time, the right to substituted performance lapses. The costs of substituted 
performance are recoverable as if they were damages.

Practical impact: Before 2018, a developer who failed to deliver a flat was often 
only liable for damages. Post-amendment, the buyer can compel delivery of the 
specific flat (specific performance as a right) or arrange alternative completion 
and recover all costs. This is a significant shift in real estate and infrastructure 
contract disputes.
        """
    }
]


def init_llm():
    google_key = os.environ.get("GOOGLE_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")

    if google_key and _GEMINI_AVAILABLE:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=google_key
        )
    else:
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            groq_api_key=groq_key
        )


def init_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')


def init_chromadb(embedder):
    client = chromadb.Client()
    try:
        client.delete_collection("legal_kb")
    except Exception:
        pass

    collection = client.create_collection(
        name="legal_kb",
        metadata={"hnsw:space": "cosine"}
    )

    texts = [doc["text"].strip() for doc in DOCUMENTS]
    ids = [doc["id"] for doc in DOCUMENTS]
    topics = [doc["topic"] for doc in DOCUMENTS]
    metadatas = [{"topic": t} for t in topics]
    embeddings = embedder.encode(texts).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    return collection


def make_memory_node(llm):
    def memory_node(state: CapstoneState) -> dict:
        question = state["question"]
        messages = state.get("messages", [])

        messages = messages + [{"role": "user", "content": question}]
        messages = messages[-6:]

        user_name = state.get("user_name", "")
        name_match = re.search(r"my name is ([A-Za-z]+)", question, re.IGNORECASE)
        if name_match:
            user_name = name_match.group(1).capitalize()

        if not user_name:
            for msg in messages:
                if msg.get("role") == "user":
                    past_match = re.search(r"my name is ([A-Za-z]+)", msg.get("content", ""), re.IGNORECASE)
                    if past_match:
                        user_name = past_match.group(1).capitalize()
                        break

        doc_type = state.get("document_type", "general")
        keyword_map = {
            "contract": "contract_law",
            "nda": "non_disclosure",
            "employment": "employment_law",
            "bail": "criminal_law",
            "arbitration": "arbitration",
            "consumer": "consumer_law",
            "evidence": "evidence_law",
            "limitation": "limitation_act",
            "power of attorney": "poa",
            "fraud": "corporate_fraud",
            "injunction": "specific_relief",
            "specific performance": "specific_relief",
            "confidential": "confidentiality",
        }
        q_lower = question.lower()
        for keyword, dtype in keyword_map.items():
            if keyword in q_lower:
                doc_type = dtype
                break

        return {
            "messages": messages,
            "user_name": user_name,
            "document_type": doc_type,
            "eval_retries": state.get("eval_retries", 0),
        }

    return memory_node


def make_router_node(llm):
    def router_node(state: CapstoneState) -> dict:
        question = state["question"]
        history_text = ""
        for msg in state.get("messages", [])[-4:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            history_text += f"{role.upper()}: {content}\n"

        prompt = f"""You are a router for a Legal Document Assistant chatbot.

Given the user's question, decide which route to take. Reply with EXACTLY ONE WORD only.

Routes:
- retrieve: Use this for questions about legal concepts, sections, acts, case law, 
  contract clauses, court procedures, bail, arbitration, evidence, consumer law, 
  corporate fraud, NDA, employment law, power of attorney, or any legal topic.
- tool: Use this ONLY when the question requires calculating a date, deadline, 
  limitation period expiry, notice period end date, or any date arithmetic 
  (e.g., "when does my limitation period expire", "calculate the deadline", 
  "2 years from March 2022 when does it end").
- memory_only: Use this ONLY for pure greetings ("hello", "hi", "thanks", 
  "okay"), one-word replies, or when the user asks what was said earlier.
  If the question references ANY legal topic — even indirectly or as a 
  follow-up to a prior legal discussion — route to retrieve, NOT memory_only.

Critical rule: If the conversation history contains a legal topic (contract, 
frustration, Section 56, bail, NDA, arbitration, etc.) and the current 
question is a follow-up about that topic (e.g. "Can X apply here?", 
"What about Y?", "Does this affect Z?"), ALWAYS route to retrieve.
When in doubt between retrieve and memory_only, choose retrieve.

Conversation history:
{history_text}

Current question: {question}

Reply with ONE WORD only — retrieve, tool, or memory_only:"""

        response = llm.invoke([HumanMessage(content=prompt)])
        route = response.content.strip().lower().split()[0]

        if route not in ["retrieve", "tool", "memory_only"]:
            route = "retrieve"

        return {"route": route}

    return router_node


def make_retrieval_node(embedder, collection):
    def retrieval_node(state: CapstoneState) -> dict:
        question = state["question"]
        query_embedding = embedder.encode([question]).tolist()[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        context_parts = []
        sources = []
        for doc, meta, dist in zip(docs, metas, distances):
            topic = meta.get("topic", "Unknown")
            similarity = round(1 - dist, 3)
            context_parts.append(f"[{topic}] (similarity: {similarity})\n{doc.strip()}")
            sources.append(topic)

        retrieved = "\n\n---\n\n".join(context_parts)
        return {"retrieved": retrieved, "sources": sources}

    return retrieval_node


def make_skip_node():
    def skip_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": [], "tool_result": ""}

    return skip_node


def make_tool_node(llm):
    def tool_node(state: CapstoneState) -> dict:
        question = state["question"]

        try:
            extract_prompt = f"""Extract date calculation information from this legal question.
Return ONLY a JSON object with these exact keys:
{{
  "base_date": "YYYY-MM-DD or null",
  "period_years": integer or 0,
  "period_months": integer or 0,
  "period_days": integer or 0,
  "acknowledgement_date": "YYYY-MM-DD or null",
  "notice_period_days": integer or 0,
  "calculation_type": "limitation / notice / acknowledgement / general"
}}

Rules:
- For Section 80 CPC legal notices, notice_period_days MUST be 60. Never return 0.
- If the question mentions "legal notice" or "Section 80", set calculation_type to "notice" and notice_period_days to 60.

Question: {question}

Return ONLY valid JSON, nothing else:"""

            response = llm.invoke([HumanMessage(content=extract_prompt)])
            raw = response.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()

            data = json.loads(raw)

            base_date_str = data.get("base_date")
            ack_date_str = data.get("acknowledgement_date")
            calc_type = data.get("calculation_type", "general")

            result_lines = []

            if ack_date_str and ack_date_str != "null":
                ack_date = datetime.strptime(ack_date_str, "%Y-%m-%d")
                years = data.get("period_years", 3)
                new_deadline = ack_date + relativedelta(years=years)
                result_lines.append(
                    f"Under Section 18 of the Limitation Act 1963, the written "
                    f"acknowledgement dated {ack_date.strftime('%B %d, %Y')} resets "
                    f"the limitation period. Fresh limitation period of {years} year(s) "
                    f"runs from the acknowledgement date."
                )
                result_lines.append(
                    f"New limitation deadline: {new_deadline.strftime('%B %d, %Y')}"
                )

            elif base_date_str and base_date_str != "null":
                base_date = datetime.strptime(base_date_str, "%Y-%m-%d")

                _llm_days = int(data.get("notice_period_days") or 0)
                if calc_type == "notice" or _llm_days > 0:
                    days = _llm_days if _llm_days > 0 else 60
                    suit_date = base_date + timedelta(days=days)
                    result_lines.append(f"Legal notice served on: {base_date.strftime('%B %d, %Y')}")
                    result_lines.append(f"Notice period: {days} days")
                    result_lines.append(f"Earliest date to file suit: {suit_date.strftime('%B %d, %Y')}")

                else:
                    years = data.get("period_years", 0)
                    months = data.get("period_months", 0)
                    days = data.get("period_days", 0)
                    deadline = base_date + relativedelta(years=years, months=months, days=days)
                    period_desc = []
                    if years:
                        period_desc.append(f"{years} year(s)")
                    if months:
                        period_desc.append(f"{months} month(s)")
                    if days:
                        period_desc.append(f"{days} day(s)")

                    result_lines.append(f"Start date: {base_date.strftime('%B %d, %Y')}")
                    result_lines.append(f"Limitation period: {', '.join(period_desc)}")
                    result_lines.append(f"Deadline (last date to file): {deadline.strftime('%B %d, %Y')}")
                    result_lines.append(f"Today's date: {datetime.now().strftime('%B %d, %Y')}")
                    days_remaining = (deadline - datetime.now()).days
                    if days_remaining > 0:
                        result_lines.append(f"Days remaining: {days_remaining} days")
                    else:
                        result_lines.append(
                            f"WARNING: Limitation period has EXPIRED by "
                            f"{abs(days_remaining)} days. Consider Section 5 "
                            f"condonation of delay (available for appeals and "
                            f"applications, NOT suits)."
                        )

            else:
                result_lines.append(
                    "Could not extract a valid date from the question. "
                    "Please provide a specific date (e.g., 'limitation period of "
                    "3 years starting from January 15, 2022')."
                )

            tool_result = "\n".join(result_lines)
            return {"tool_result": tool_result, "retrieved": "", "sources": []}

        except Exception as e:
            error_msg = (
                f"Date calculation encountered an error: {str(e)}. "
                f"Please specify the date clearly (e.g., 'starting from March 1, 2022, "
                f"with a 3-year limitation period')."
            )
            return {"tool_result": error_msg, "retrieved": "", "sources": []}

    return tool_node


def make_answer_node(llm):
    def answer_node(state: CapstoneState) -> dict:
        question = state["question"]
        retrieved = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")
        eval_retries = state.get("eval_retries", 0)
        user_name = state.get("user_name", "")
        messages = state.get("messages", [])

        history_text = ""
        for msg in messages[-4:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            history_text += f"{role.upper()}: {content}\n"

        grounding_instruction = (
            "STRICT WARNING: This is a retry because the previous answer was not "
            "sufficiently grounded in the provided context. You MUST use ONLY the "
            "information provided below. Do not add ANY external legal knowledge."
            if eval_retries > 0
            else ""
        )

        name_prefix = f"(The user's name is {user_name}. Address them by name where natural.) " if user_name else ""

        system_prompt = f"""You are a Legal Document Assistant for paralegals and junior lawyers in India.
{name_prefix}
{grounding_instruction}

IMPORTANT RULES:
1. Answer ONLY using the context provided below — KNOWLEDGE BASE CONTEXT and TOOL RESULT.
2. NEVER fabricate legal section numbers, case names, or provisions not present in the context.
3. If the context does not contain enough information to answer, say clearly: 
   "I don't have enough information in my knowledge base to answer this accurately. 
   Please consult a qualified advocate or refer to the primary source."
4. Never predict court outcomes or give advice as a substitute for legal counsel.
5. For emergency or urgent legal matters, always recommend consulting a qualified advocate immediately.
6. Respond with India-specific legal context unless the question specifies another jurisdiction.

=== KNOWLEDGE BASE CONTEXT ===
{retrieved if retrieved else "No knowledge base context retrieved for this query."}

=== TOOL RESULT ===
{tool_result if tool_result else "No tool calculation was performed for this query."}

=== CONVERSATION HISTORY ===
{history_text}

=== USER QUESTION ===
{question}

Provide a clear, structured, accurate legal answer based ONLY on the context above:"""

        response = llm.invoke([HumanMessage(content=system_prompt)])
        answer = response.content.strip()
        return {"answer": answer}

    return answer_node


def make_eval_node(llm):
    FAITHFULNESS_THRESHOLD = 0.7
    MAX_EVAL_RETRIES = 2

    def eval_node(state: CapstoneState) -> dict:
        retrieved = state.get("retrieved", "")
        answer = state.get("answer", "")
        eval_retries = state.get("eval_retries", 0)

        if not retrieved.strip():
            return {"faithfulness": 1.0, "eval_retries": eval_retries}

        eval_prompt = f"""You are evaluating an AI legal assistant's answer for faithfulness.

FAITHFULNESS means: does the answer contain ONLY information present in the 
retrieved context below? If the answer introduces legal provisions, case names, 
or facts NOT found in the retrieved context, faithfulness is LOW.

Retrieved Context:
{retrieved[:2000]}

Answer to Evaluate:
{answer[:1500]}

Score the faithfulness from 0.0 to 1.0:
- 1.0: Every claim in the answer is directly supported by the retrieved context
- 0.7-0.9: Most claims supported, minor additions from general knowledge
- 0.4-0.6: Some claims not in retrieved context — answer adds external knowledge
- 0.0-0.3: Answer largely ignores context and uses training knowledge

Reply with ONLY a decimal number between 0.0 and 1.0 (e.g., 0.85):"""

        response = llm.invoke([HumanMessage(content=eval_prompt)])
        try:
            score = float(re.search(r"\d+\.?\d*", response.content.strip()).group())
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.5

        new_retries = eval_retries + 1
        return {"faithfulness": score, "eval_retries": new_retries}

    return eval_node


def make_save_node():
    def save_node(state: CapstoneState) -> dict:
        messages = state.get("messages", [])
        answer = state.get("answer", "")
        messages = messages + [{"role": "assistant", "content": answer}]
        messages = messages[-6:]
        return {"messages": messages}

    return save_node


FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2


def route_decision(state: CapstoneState) -> str:
    route = state.get("route", "retrieve")
    if route == "tool":
        return "tool"
    elif route == "memory_only":
        return "skip"
    else:
        return "retrieve"


def eval_decision(state: CapstoneState) -> str:
    faithfulness = state.get("faithfulness", 1.0)
    eval_retries = state.get("eval_retries", 0)

    if faithfulness < FAITHFULNESS_THRESHOLD and eval_retries < MAX_EVAL_RETRIES:
        return "answer"

    return "save"


def build_graph(groq_api_key: str = None, gemini_api_key: str = None):
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key

    llm = init_llm()
    embedder = init_embedder()
    collection = init_chromadb(embedder)

    memory_node = make_memory_node(llm)
    router_node = make_router_node(llm)
    retrieval_node = make_retrieval_node(embedder, collection)
    skip_node = make_skip_node()
    tool_node = make_tool_node(llm)
    answer_node = make_answer_node(llm)
    eval_node = make_eval_node(llm)
    save_node = make_save_node()

    graph = StateGraph(CapstoneState)

    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip", skip_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    graph.set_entry_point("memory")

    graph.add_edge("memory", "router")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip", "answer")
    graph.add_edge("tool", "answer")
    graph.add_edge("answer", "eval")
    graph.add_edge("save", END)

    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve": "retrieve",
            "tool": "tool",
            "skip": "skip"
        }
    )

    graph.add_conditional_edges(
        "eval",
        eval_decision,
        {
            "answer": "answer",
            "save": "save"
        }
    )

    app = graph.compile(checkpointer=MemorySaver())
    return app, embedder, collection, llm


def ask(question: str, thread_id: str, app) -> dict:
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "question": question,
        "route": "",
        "retrieved": "",
        "sources": [],
        "tool_result": "",
        "answer": "",
        "faithfulness": 0.0,
        "eval_retries": 0,
        "document_type": "general"
    }

    result = app.invoke(initial_state, config=config)

    return {
        "answer": result.get("answer", ""),
        "route": result.get("route", ""),
        "faithfulness": result.get("faithfulness", 0.0),
        "sources": result.get("sources", []),
        "eval_retries": result.get("eval_retries", 0),
        "user_name": result.get("user_name", ""),
        "document_type": result.get("document_type", "general")
    }


if __name__ == "__main__":
    api_key = input("Enter GROQ_API_KEY: ").strip()
    app, embedder, collection, llm = build_graph(groq_api_key=api_key)

    test_q = "What is the difference between a void and voidable contract under ICA?"
    result = ask(test_q, thread_id="smoke_test_001", app=app)
    print(f"\nQ: {test_q}")
    print(f"Route: {result['route']}")
    print(f"Faithfulness: {result['faithfulness']:.2f}")
    print(f"Answer:\n{result['answer']}")