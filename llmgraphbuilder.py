import streamlit as st
from pyvis.network import Network
from neo4j import GraphDatabase
import pandas as pd
from langchain.llms import AI21
from langchain.chains import LLMChain, PromptTemplate
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInference

# Set up Neo4j connection
uri = "bolt://localhost:7687"  # Adjust if using a different host/port
username = "neo4j"
password = "password"  # Replace with your password

# Connect to the Neo4j database
driver = GraphDatabase.driver(uri, auth=(username, password))

# Function to fetch graph data
def fetch_graph_data(query):
    with driver.session() as session:
        results = session.run(query)
        nodes = []
        edges = []
        for record in results:
            n = record["n"]
            m = record["m"]
            r = record["r"]
            nodes.append(n)
            nodes.append(m)
            edges.append((n.id, m.id, r.type))
        # Removing duplicates
        nodes = {n.id: n for n in nodes}.values()
        return nodes, edges

# Function to visualize graph
def visualize_graph(nodes, edges):
    net = Network(height="750px", width="100%", notebook=True)
    for node in nodes:
        net.add_node(node.id, label=str(node.id), title=node.labels)
    for edge in edges:
        net.add_edge(edge[0], edge[1], label=edge[2])
    net.show("graph.html")
    HtmlFile = open("graph.html", "r", encoding="utf-8")
    source_code = HtmlFile.read()
    return source_code

# Set up LLM for natural language queries
llm = AI21()

# Define a prompt template for querying the database
template = PromptTemplate(
    input_variables=["query"],
    template="Query the Neo4j database with the following Cypher query: {query}",
)

# Define a chain to generate Cypher queries
chain = LLMChain(
    llm=llm,
    prompt=template,
)

# Streamlit app
st.title("Neo4j Database Query and Visualization")

# User input for query
query_input = st.text_input("Enter your query")

if st.button("Submit"):
    try:
        # Generate Cypher query using LLM
        cypher_query = chain({"query": query_input})
        st.write(f"Generated Cypher Query: {cypher_query}")
        
        # Execute Cypher query against Neo4j
        nodes, edges = fetch_graph_data(cypher_query)
        
        # Visualize the graph
        graph_html = visualize_graph(nodes, edges)
        st.components.v1.html(graph_html, height=800, width=1000)
        
    except Exception as e:
        st.error(f"Error processing query: {e}")

# Close Neo4j driver when done
def close_driver():
    driver.close()

# Close driver on app exit
import atexit
atexit.register(close_driver)
