import streamlit as st
from pyvis.network import Network
from neo4j import GraphDatabase
import pandas as pd
from langchain.llms import OpenAI
from langchain.chains import LLMChain, PromptTemplate

# Fetch credentials from Streamlit secrets
URI = st.secrets["NEO4J_URI"]  # Example: "neo4j+s://b13c8ca5.databases.neo4j.io"
AUTH = (st.secrets["NEO4J_USERNAME"], st.secrets["NEO4J_PASSWORD"])  # Username and password
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Connect to the Neo4j Aura database
try:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()
    st.success("Connected to Neo4j Aura database successfully!")
except Exception as e:
    st.error(f"Failed to connect to Neo4j Aura database: {e}")

# Function to create nodes and relationships
def create_graph_data(query):
    with driver.session() as session:
        # Generate Cypher query to create nodes and relationships
        cypher_query = f"""
            UNWIND split('{query}', ',') AS entity
            MERGE (n:Entity {{name: trim(entity)}})
        """
        session.run(cypher_query)
        
        # Example to create relationships (adjust based on your needs)
        relationship_query = """
            MATCH (n1:Entity {name: 'Entity1'}), (n2:Entity {name: 'Entity2'})
            MERGE (n1)-[:RELATED_TO]->(n2)
        """
        session.run(relationship_query)

# Function to fetch graph data
def fetch_graph_data(query):
    with driver.session() as session:
        results = session.run(query)
        nodes = []
        edges = []
        for record in results:
            if "n" in record and "m" in record and "r" in record:
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

# Set up LLM for natural language queries using OpenAI GPT-4
llm = OpenAI(model_name="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)

# Define a prompt template for querying the database
template = PromptTemplate(
    input_variables=["query"],
    template="Convert the following text into Cypher queries to create nodes and relationships: {query}",
)

# Define a chain to generate Cypher queries
chain = template | llm

# Streamlit app
st.title("Neo4j Database Query and Visualization")

# User input for query
query_input = st.text_input("Enter your query")

if st.button("Submit"):
    try:
        # Generate Cypher query using LLM
        cypher_query = chain({"query": query_input})
        st.write(f"Generated Cypher Query: {cypher_query}")
        
        # Create nodes and relationships
        create_graph_data(query_input)
        
        # Fetch graph data
        fetch_query = "MATCH (n)-[r]->(m) RETURN n, m, r"
        nodes, edges = fetch_graph_data(fetch_query)
        
        # Visualize the graph
        if nodes and edges:
            graph_html = visualize_graph(nodes, edges)
            st.components.v1.html(graph_html, height=800, width=1000)
        else:
            st.info("No nodes or edges found in the query result.")
        
    except Exception as e:
        st.error(f"Error processing query: {e}")

# Close Neo4j driver when done
def close_driver():
    driver.close()

# Close driver on app exit
import atexit
atexit.register(close_driver)
