# ------------------------------------------------------------------
# 4) LANGGRAPH AGENT
# ------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Dict, Any
from retrieval_system import FinancialRetrievalSystem, RetrievalResult, RetrievalConfig
import os, logging

class AgentState(TypedDict):
    query: str
    docs: List[RetrievalResult]
    notes: List[str]
    data: Dict[str, Any]
    answer: str

# -------- Tools ---------------------------------------------------
class DocSearchTool(BaseTool):
    name: str = "doc_search"
    description: str = "Semantic search over JP disclosures."
    
    def __init__(self, sys: FinancialRetrievalSystem): 
        super().__init__()
        # Store the system reference in a way that doesn't conflict with Pydantic
        object.__setattr__(self, '_sys', sys)
    
    async def _arun(self, q: str, filters: dict = None): 
        return await self._sys.search(q, filters=filters or {}, k=25)
    
    def _run(self, q: str, filters: dict = None): 
        raise NotImplementedError("Use async version _arun instead")

class Agent:
    def __init__(self, sys: FinancialRetrievalSystem, llm: ChatOpenAI):
        self.sys, self.llm = sys, llm
        self.search = DocSearchTool(sys)
        self.flow = self._wire()

    def _wire(self):
        g = StateGraph(AgentState)
        g.add_node("search", self._n_search)
        g.add_node("compose", self._n_compose)
        g.set_entry_point("search")
        g.add_edge("search", "compose"); g.add_edge("compose", END)
        return g.compile()

    # ---- nodes ----------------------------------------------------
    async def _n_search(self, st: AgentState) -> AgentState:
        st.setdefault("notes", []).append("🔎 running dense/colbert search")
        st["docs"] = await self.search._arun(st["query"])
        st["notes"].append(f"found {len(st['docs'])} docs")
        return st

    async def _n_compose(self, st: AgentState) -> AgentState:
        ctx_docs = "\n".join(f"- {d.date} {d.name} {d.title} (score {d.score:.2f})" for d in st["docs"][:8])
        prompt = f"""You are a bilingual financial analyst (JA/EN).
User question: "{st['query']}"

Top evidence:
{ctx_docs or 'NO HITS'}

Respond in English, concise bullet-points, cite company names + dates."""
        st["answer"] = (await self.llm.ainvoke(prompt)).content
        return st

    # ---- external API --------------------------------------------
    async def run(self, q: str) -> str:
        init = AgentState(query=q, docs=[], notes=[], data={}, answer="")
        final = await self.flow.ainvoke(init)
        return final["answer"]

# ------------------------------------------------------------------
# 5) QUICK SELF-TEST (comment out in prod)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio, sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s – %(message)s")
    
    cfg = RetrievalConfig(
        pg_dsn=os.getenv("PG_DSN", "postgresql://user:pass@localhost/financial_db"),
        redis_url=os.getenv("REDIS_URL","redis://localhost:6379/0")
    )
    
    async def _demo():
        sys = FinancialRetrievalSystem(cfg); await sys.init()
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        agent = Agent(sys, llm)
        print(await agent.run("What companies revised their earnings guidance upward recently?"))
        await sys.close()
    asyncio.run(_demo())
