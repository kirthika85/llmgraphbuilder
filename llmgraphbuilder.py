import streamlit as st
from pyvis.network import Network
from neo4j import GraphDatabase
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Fetch credentials from Streamlit secrets
URI = st.secrets["NEO4J_URI"]  # Example: "neo4j+s://your-instance.databases.neo4j.io"
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
def create_graph_data(cypher_query):
    st.write(f"Creating nodes and relationships using query: {cypher_query}")
    with driver.session() as session:
        try:
            st.write(f"Executing Cypher query: {cypher_query}")
            session.run(cypher_query)
            st.write("Nodes and relationships created successfully.")
        except Exception as e:
            st.error(f"Error executing Cypher query: {e}")

# Function to fetch graph data dynamically
def fetch_graph_data():
    try:
        with driver.session() as session:
            # Fetch all nodes and their relationships dynamically
            query = """
                MATCH (n)-[r]->(m)
                RETURN n, r, m LIMIT 100
            """
            st.write(f"Executing Cypher query: {query}")
            results = session.run(query)
            
            nodes = []
            edges = []
            
            for record in results:
                n = record["n"]
                m = record["m"]
                r = record["r"]
                
                # Convert frozenset properties to lists for JSON serialization
                n_properties = {k: list(v) if isinstance(v, frozenset) else v for k, v in dict(n).items()}
                m_properties = {k: list(v) if isinstance(v, frozenset) else v for k, v in dict(m).items()}
                
                nodes.append({"id": n.id, "properties": n_properties})
                nodes.append({"id": m.id, "properties": m_properties})
                edges.append((n.id, m.id, r.type))
            
            # Remove duplicate nodes based on their IDs
            unique_nodes = {node["id"]: node for node in nodes}.values()
            
            st.write(f"Fetched {len(unique_nodes)} unique nodes and {len(edges)} edges.")
            return unique_nodes, edges
            
    except Exception as e:
        st.error(f"Error fetching graph data: {e}")
        return [], []

# Function to visualize graph using PyVis
def visualize_graph(nodes, edges):
    st.write("Visualizing graph...")
    net = Network(height="750px", width="100%", notebook=True)
    
    for node in nodes:
        net.add_node(node["id"], label=str(node["properties"].get("name", "Unnamed")), title=str(node["properties"]))
    
    for edge in edges:
        net.add_edge(edge[0], edge[1], label=edge[2])
    
    net.show("graph.html")
    HtmlFile = open("graph.html", "r", encoding="utf-8")
    source_code = HtmlFile.read()
    st.write("Graph visualization complete.")
    return source_code

# Set up LLM for natural language queries using OpenAI GPT-4
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)

# Define a prompt template for generating Cypher queries
template = PromptTemplate(
    input_variables=["query"],
    template="Convert the following text into Cypher queries to create nodes and relationships: {query}"
)

# Streamlit app setup
st.title("Neo4j Database Query and Visualization")

# User input for query
query_input = st.text_input("Enter your query")

if st.button("Submit"):
    try:
        # Generate Cypher query using LLM
        prompt = template.format(query=query_input)
        st.write(f"Prompt: {prompt}")
        response = llm([HumanMessage(content=prompt)])
        
        # Access the response content directly (the generated Cypher query)
        cypher_query = response.content
        
        st.write(f"Generated Cypher Query: {cypher_query}")
        
        # Create nodes and relationships in Neo4j database
        create_graph_data(cypher_query)
        
        # Fetch graph data dynamically for visualization
        nodes, edges = fetch_graph_data()
        
        # Visualize the graph using PyVis if data exists
        if nodes or edges:
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
