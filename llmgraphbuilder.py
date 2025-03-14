import streamlit as st
import os
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Fetch credentials from environment variables
URI = os.getenv("NEO4J_URI")  # Example: "neo4j+s://your-instance.databases.neo4j.io"
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Connect to Neo4j Aura database
driver = GraphDatabase.driver(URI, auth=AUTH)

# Set up LLM for natural language queries using OpenAI GPT-4
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)

# Define a prompt template for generating Cypher queries dynamically
template = PromptTemplate(
    input_variables=["query"],
    template="""
    Convert the following text into Cypher queries to create nodes and relationships:
    - Use "CAPITAL" for relationships between countries and their capitals.
    - Use "CONTAINS" for relationships between solar systems and planets.
    - Use "HAS_DEFINITION" for mapping topics like "Artificial Intelligence" with their definitions.
    Query: {query}
    """
)

# Function to convert natural language to Cypher query using LLM
def convert_to_cypher(question):
    prompt = template.format(query=question)
    response = llm([HumanMessage(content=prompt)])
    return response.content

# Function to execute Cypher query and display results
def execute_cypher_query(cypher_query):
    with driver.session() as session:
        try:
            results = session.run(cypher_query)
            print("Query Results:")
            for record in results:
                print(record)
        except Exception as e:
            print(f"Error executing Cypher query: {e}")

# Function to create nodes and relationships dynamically using Cypher query
def create_graph_data(cypher_queries):
    with driver.session() as session:
        for query in cypher_queries:
            try:
                print(f"Executing Cypher query: {query}")
                session.run(query)
                print("Nodes and relationships created successfully.")
            except Exception as e:
                print(f"Error executing Cypher query: {e}")

# Function to fetch all graph data dynamically (nodes with and without relationships)
def fetch_graph_data():
    try:
        with driver.session() as session:
            # Fetch all nodes and their relationships with properties
            query_with_relationships = """
                MATCH (n)-[r]->(m)
                RETURN n, r, m LIMIT 100
            """
            print(f"Executing Cypher query for nodes with relationships: {query_with_relationships}")
            results_with_relationships = session.run(query_with_relationships)
            
            nodes = []
            edges = []
            
            for record in results_with_relationships:
                n = record["n"]
                m = record["m"]
                r = record["r"]
                
                # Extract node properties for JSON serialization
                n_properties = {k: list(v) if isinstance(v, frozenset) else v for k, v in dict(n).items()}
                m_properties = {k: list(v) if isinstance(v, frozenset) else v for k, v in dict(m).items()}
                
                nodes.append({"id": n.id, "name": n_properties.get("name", "Unnamed"), "properties": n_properties})
                nodes.append({"id": m.id, "name": m_properties.get("name", "Unnamed"), "properties": m_properties})
                edges.append({"source": n.id, "target": m.id, "type": r.type})
            
            # Fetch all nodes without relationships
            query_without_relationships = """
                MATCH (n)
                WHERE NOT (n)--()
                RETURN n LIMIT 100
            """
            print(f"Executing Cypher query for nodes without relationships: {query_without_relationships}")
            results_without_relationships = session.run(query_without_relationships)
            
            for record in results_without_relationships:
                n = record["n"]
                
                # Extract node properties for JSON serialization
                n_properties = {k: list(v) if isinstance(v, frozenset) else v for k, v in dict(n).items()}
                
                nodes.append({"id": n.id, "name": n_properties.get("name", "Unnamed"), "properties": n_properties})
            
            # Remove duplicate nodes based on their IDs
            unique_nodes = {node["id"]: node for node in nodes}.values()
            
            print(f"Fetched {len(unique_nodes)} unique nodes and {len(edges)} edges.")
            return unique_nodes, edges
            
    except Exception as e:
        print(f"Error fetching graph data: {e}")
        return [], []

# Streamlit app setup
st.title("Neo4j Database Query and Visualization")

# User input for query
question = st.text_area("Enter your question")

if st.button("Submit"):
    try:
        # Convert question to Cypher query using LLM
        cypher_query_text = convert_to_cypher(question)
        print("Generated Cypher Query:", cypher_query_text)
        
        # Extract actual Cypher queries from the response
        cypher_queries = [line.strip() for line in cypher_query_text.splitlines() if line.strip().startswith(("CREATE", "MATCH"))]
        print("Extracted Cypher Queries:", cypher_queries)
        
        # Create nodes and relationships in Neo4j database dynamically
        create_graph_data(cypher_queries)
        
        # Fetch graph data dynamically for visualization or debugging purposes
        nodes, edges = fetch_graph_data()
        
        print("\nGraph Data:")
        print(f"Nodes: {nodes}")
        print(f"Edges: {edges}")
        
    except Exception as e:
        print(f"Error processing query: {e}")

# Close Neo4j driver when done
def close_driver():
    driver.close()

# Close driver on app exit
import atexit
atexit.register(close_driver)
