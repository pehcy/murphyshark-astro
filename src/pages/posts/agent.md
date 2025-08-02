---
layout: ../../layouts/MarkdownPostLayout.astro
title: Agent with Langchain
author: pehcy
description: "Creates an Q&A assistant with Gemini-2.0 and Langchain"
image:
    url: "https://docs.astro.build/assets/arc.webp"
    alt: ""
pubDate: 2025-05-09
tags: ["agent", "LLM", "🤗"]
---

<h1 class="gradient-text">Agent with Langchain</h1>


## What do agent used for?

Think of the Agent as a person, it consists of two main parts.
1. Brain (LLM, AI Model), this is the main core of agent where it handles reasoning and planning.
2. The Body (Tools), this is the part where interact with its environment. It received the strategies and plans from LLM and performing the possible actions that it equipped with. 

Now let's us build a little agent with `langchain`. 

## Toy Example

The metadata.jsonl recorded all the example questions and its corresponding correct answer for retrieval.
```python
# Retrieve first 30 rows of GAIA questions
sample = qa_lines[30]
# Tools required to answer the first 30 questions
sample['Annotator Metadata']['Tools'].split('\n')
# >>> ['1. Search engine', '2. Web browser', '3. PDF viewer']
```
We can now see that to answer these types of questions, the agent need to search the answers online via search engine (Google, Duckduckgo, Yahoo, etc.) through any web browser (Firefox, Edge, Google Chrome, etc.), as well as PDF viewer to look through available Arxiv articles or journal.
These are the basic toolset that we need to provide for our homebrew agent:

1. Calculator, provided basic tools for our agent to perform basic mathematical calculations.
2. Web search, this allow our agent to retrieve latest informations through search engine.
3. Arxiv, so that our agent can summarize the criteria from published Arxiv papers.
4. Retrieval system, this will determine the properties of the given question, and hence looking for similar questions from the database with similar embedding. The similarity of the embeddings is measured with cosine distance.

#### Step 1. Setup Question Bank database

In SQL Editor, copy the following code snippet to create a table called `document` to store the questions and its embedding vector.
```sql
-- Create a table to store your documents
create table documents (
	id bigserial primary key,
	content text, -- corresponds to Document.pageContent
	metadata jsonb, -- corresponds to Document.metadata
	embedding vector(768) -- 768 works for Gemini embeddings, change if needed
);
```

Before that, you may need to proceed to `Extension` settings in your Supabase dashboard. You need to have pgvectors extension installed and make sure it is enabled.

```sql
create function match_documents (
	query_embedding vector(768),
	match_count int DEFAULT null,
	filter jsonb DEFAULT '{}'
) returns table (
	id bigint,
	content text,
	metadata jsonb,
	similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
	return query 
	select 
		id, 
		content, 
		metadata, 
		1 - (documents.embedding <=> query_embedding) as similarity
	from documents
	where metadata @> filter
	order by documents.embedding <=> query_embedding
	limit match_count;
end;
$$;
```

#### Step 2. Create calculator.
We begin with some simple mathematical calculator tools.
```python
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers.
    Args:
        a: first int
        b: second int
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.
    Args:
        a: first int
        b: second int
    """
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.
    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def divide(a: int, b: int) -> int:
    """Divide first number by second number.
    Args:
        a: first int
        b: second int
    """
    try:
        return a / b
    except ZeroDivisionError:
        return None

@tool
def power(a: int, b: int) -> int:
    """Power up first number by second number.
    Args:
        a: first int
        b: second int
    """
    return a ** b

@tool
def modulus(a: int, b: int) -> int:
    """Get remainder of first number divided by second number.
    Args:
        a: first int
        b: second int
    """
    return a % b
```

```python
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.tools import tool
+ from langchain_community.tools.tavily_search import TavilySearchResults
+ from langchain_community.document_loaders import WikipediaLoader
+ from langchain_community.document_loaders import ArxivLoader

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.
    Args:
        query: The search query.
    """
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join([
        f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n\t{doc.page_content}\n<Document>'
        for doc in search_docs
    ])
    return { "wiki_results": formatted_search_docs }

@tool
def web_search(query: str) -> str:
    """Search Tavily for a query and return maximum 3 results.
    Args:
        query: The search query.
    """
    search_docs = TavilySearchResults(max_results=3).invoke(input=query)
    formatted_search_docs = "\n\n---\n\n".join([
        f'<Document source="{doc["url"]}"/>\n\t{doc["content"]}\n<Document>'
        for doc in search_docs
    ])
    return { "web_results": formatted_search_docs }

@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    Args:
        query: The search query.
    """
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join([
        f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n\t{doc.page_content[:1000]}\n<Document>'
        for doc in search_docs
    ])
    return { "arxiv_results": formatted_search_docs }

# list of tools

tools = [
    add,
    subtract,
    multiply,
    divide,
    power,
    modulus,
    wiki_search,
    web_search,
    arxiv_search
]
```

```python
# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```
Note that all the langchain toolset function must begin with `@tool`.

#### Step 3: Build graph
```python
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
# llm = ChatGroq(model="qwen-qwq-32b", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# Node
def assistant(state: AgentState):
	"""Assistant node"""
	return { "messages": [llm_with_tools.invoke(state['messages'])] }

def retriever(state: AgentState):
        similar_question = vector_store.similarity_search(state['messages'][0].content)
        example_msg = HumanMessage(
            content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
        )
	return { "messages": [__sys_msg] + state['messages'] + [example_msg] }



builder = StateGraph(AgentState)
# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("retriever", retriever)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "retriever")
builder.add_conditional_edges(
	"assistant",
    tools_condition
)
builder.add_edge("tools", "assistant")
builder.add_edge("retriever", "assistant")
```
![[Pasted image 20250720030152.png]]

#### Prompt
```txt
You are a helpful agent responsible for answering questions using a set of tools provided.
If the tool(s) not available, you can try to search and find the solution or information online.
You can also use your own knowledge to answer the question.
==========================
Here is a few examples showing you how to answer the question step by step.

Question 1: In terms of geographical distance between capital cities, which 2 countries are the furthest from each other within the ASEAN bloc according to wikipedia? Answer using a comma separated list, ordering the countries by alphabetical order.
Steps:
1. Search the web for "ASEAN bloc".
2. Click the Wikipedia result for the ASEAN Free Trade Area.
3. Scroll down to find the list of member states.
4. Click into the Wikipedia pages for each member state, and note its capital.
5. Search the web for the distance between the first two capitals. The results give travel distance, not geographic distance, which might affect the answer.
6. Thinking it might be faster to judge the distance by looking at a map, search the web for "ASEAN bloc" and click into the images tab.
7. View a map of the member countries. Since they're clustered together in an arrangement that's not very linear, it's difficult to judge distances by eye.
8. Return to the Wikipedia page for each country. Click the GPS coordinates for each capital to get the coordinates in decimal notation.
9. Place all these coordinates into a spreadsheet.
10. Write formulas to calculate the distance between each capital.
11. Write formula to get the largest distance value in the spreadsheet.
12. Note which two capitals that value corresponds to: Jakarta and Naypyidaw.
13. Return to the Wikipedia pages to see which countries those respective capitals belong to: Indonesia, Myanmar.
Tools:
14. Search engine
15. Web browser
16. Microsoft Excel / Google Sheets
Final Answer: Indonesia, Myanmar

Question 2: Review the chess position provided in the image. It is black's turn. Provide the correct next move for black which guarantees a win. Please provide your response in algebraic notation.
Steps:
Step 1: Evaluate the position of the pieces in the chess position
Step 2: Report the best move available for black: "Rd5"
Tools:
1. Image recognition tools
Final Answer: Rd5

Question 3: Solve the equation x^2 + 5x = -6
Steps:
Step 1: Moving all terms to left-hand side until the right-hand side become zero.
Step 2: Identify the highest power of polynomial in left-hand side. In this case the highest power is 2, this equation is a quadratic equation.
Step 3: Identify the coefficients of each term in this quadratic equation.
Step 3: Write quadratic formula and calculate the possible solutions.
Tools:
1. Search engine
2. Web browser
3. Calculator 
Final Answer: x=-2, x=-3

==========================
Now, please answer the following question step by step.
```
# Reference
1. [Creating a RAG Tool for Guest Stories](https://huggingface.co/learn/agents-course/unit3/agentic-rag/invitees)