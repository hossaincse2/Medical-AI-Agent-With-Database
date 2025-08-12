import os
import sqlite3
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
import requests
import json
from datetime import datetime
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATABASE_FOLDER'] = 'databases'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATABASE_FOLDER'], exist_ok=True)

class CSVToSQLiteConverter:
    """Converts CSV files to SQLite databases"""
    
    @staticmethod
    def convert_csv_to_sqlite(csv_path, db_name):
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Create database path
            db_path = os.path.join(app.config['DATABASE_FOLDER'], f"{db_name}.db")
            
            # Connect to SQLite database
            conn = sqlite3.connect(db_path)
            
            # Convert DataFrame to SQLite table
            table_name = db_name.lower().replace('_', '')
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            conn.close()
            return True, f"Successfully converted {csv_path} to {db_path}"
        except Exception as e:
            return False, f"Error converting CSV: {str(e)}"

class DatabaseQueryTool:
    """Base class for database query tools"""
    
    def __init__(self, db_path, table_name):
        self.db_path = db_path
        self.table_name = table_name
    
    def execute_query(self, query):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            conn.close()
            
            return True, {"columns": columns, "data": results}
        except Exception as e:
            return False, f"Database query error: {str(e)}"
    
    def get_schema(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({self.table_name})")
            schema = cursor.fetchall()
            conn.close()
            return schema
        except Exception as e:
            return None

class HeartDiseaseDBTool(DatabaseQueryTool):
    def __init__(self):
        db_path = os.path.join(app.config['DATABASE_FOLDER'], "heart.db")
        super().__init__(db_path, "heart")
    
    def analyze_question(self, question):
        """Convert natural language to SQL query for heart disease data"""
        question_lower = question.lower()
        
        if "average age" in question_lower or "mean age" in question_lower:
            return "SELECT AVG(age) as average_age FROM heart"
        elif "count" in question_lower and "heart disease" in question_lower:
            return "SELECT COUNT(*) as total_patients FROM heart WHERE target = 1"
        elif "cholesterol" in question_lower and "average" in question_lower:
            return "SELECT AVG(chol) as average_cholesterol FROM heart"
        elif "age distribution" in question_lower:
            return "SELECT age, COUNT(*) as count FROM heart GROUP BY age ORDER BY age"
        else:
            return f"SELECT * FROM heart LIMIT 10"

class CancerDBTool(DatabaseQueryTool):
    def __init__(self):
        db_path = os.path.join(app.config['DATABASE_FOLDER'], "cancer.db")
        super().__init__(db_path, "thecancerdata1500v2")
    
    def analyze_question(self, question):
        """Convert natural language to SQL query for cancer data"""
        question_lower = question.lower()
        
        if "average age" in question_lower:
            return "SELECT AVG(Age) as average_age FROM thecancerdata1500v2"
        elif "smoking" in question_lower and "cancer" in question_lower:
            return "SELECT Smoking, COUNT(*) as count FROM thecancerdata1500v2 WHERE Diagnosis = 1 GROUP BY Smoking"
        elif "gender distribution" in question_lower:
            return "SELECT Gender, COUNT(*) as count FROM thecancerdata1500v2 GROUP BY Gender"
        elif "bmi" in question_lower and "average" in question_lower:
            return "SELECT AVG(BMI) as average_bmi FROM thecancerdata1500v2"
        else:
            return f"SELECT * FROM thecancerdata1500v2 LIMIT 10"

class DiabetesDBTool(DatabaseQueryTool):
    def __init__(self):
        db_path = os.path.join(app.config['DATABASE_FOLDER'], "diabetes.db")
        super().__init__(db_path, "diabetes")
    
    def analyze_question(self, question):
        """Convert natural language to SQL query for diabetes data"""
        question_lower = question.lower()
        
        if "average glucose" in question_lower:
            return "SELECT AVG(Glucose) as average_glucose FROM diabetes"
        elif "diabetes count" in question_lower:
            return "SELECT COUNT(*) as diabetic_patients FROM diabetes WHERE Outcome = 1"
        elif "age distribution" in question_lower:
            return "SELECT Age, COUNT(*) as count FROM diabetes GROUP BY Age ORDER BY Age"
        elif "bmi" in question_lower and "average" in question_lower:
            return "SELECT AVG(BMI) as average_bmi FROM diabetes"
        else:
            return f"SELECT * FROM diabetes LIMIT 10"

class GitHubModelsAPI:
    """Interface for GitHub Models API using GPT-4 mini"""
    
    def __init__(self):
        self.api_key = os.getenv('GITHUB_TOKEN')
        self.endpoint = "https://models.inference.ai.azure.com"
        self.model = "gpt-4o-mini"
        
        if not self.api_key:
            raise ValueError("GITHUB_TOKEN not found in environment variables")
    
    def generate_response(self, messages, max_tokens=1000):
        """Generate response using GitHub Models API"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "model": self.model
            }
            
            response = requests.post(
                f"{self.endpoint}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return True, result['choices'][0]['message']['content']
            else:
                return False, f"API Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.Timeout:
            return False, "Request timeout - API took too long to respond"
        except requests.exceptions.RequestException as e:
            return False, f"Request error: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

class MedicalWebSearchTool:
    """Enhanced web search tool using GitHub Models API"""
    
    def __init__(self):
        self.github_api = GitHubModelsAPI()
    
    def search_medical_info(self, query):
        """Get medical information using GPT-4 mini"""
        try:
            # Create a focused medical prompt
            messages = [
                {
                    "role": "system",
                    "content": """You are a medical information assistant. Provide accurate, helpful medical information based on established medical knowledge. Always include disclaimers about consulting healthcare professionals. Focus on:
                    - Symptoms and their descriptions
                    - General treatment approaches
                    - Risk factors and prevention
                    - Basic medical definitions
                    
                    Keep responses concise but informative. Always end with advice to consult healthcare professionals for personal medical concerns."""
                },
                {
                    "role": "user", 
                    "content": f"Provide medical information about: {query}"
                }
            ]
            
            success, response = self.github_api.generate_response(messages, max_tokens=500)
            
            if success:
                return True, response
            else:
                # Fallback to basic responses if API fails
                return self._fallback_response(query)
                
        except Exception as e:
            return self._fallback_response(query)
    
    def _fallback_response(self, query):
        """Fallback responses when API is unavailable"""
        fallback_responses = {
            "heart disease": "Heart disease symptoms include chest pain, shortness of breath, and fatigue. Treatment involves lifestyle changes, medications, and sometimes surgery. Consult a cardiologist for proper diagnosis.",
            "cancer": "Cancer symptoms vary by type but may include unexplained weight loss, fatigue, and unusual lumps. Treatment options include surgery, chemotherapy, and radiation. Early detection is crucial - consult an oncologist.",
            "diabetes": "Diabetes symptoms include increased thirst, frequent urination, and fatigue. Management involves blood sugar monitoring, medication, diet, and exercise. Consult an endocrinologist for proper care."
        }
        
        for key, value in fallback_responses.items():
            if key in query.lower():
                return True, value + " (Note: API temporarily unavailable - using basic information)"
        
        return True, f"For detailed medical information about '{query}', please consult healthcare professionals or trusted medical sources. (Note: API temporarily unavailable)"

class MedicalAIAgent:
    """Enhanced AI Agent using GitHub Models for intelligent routing and responses"""
    
    def __init__(self):
        try:
            self.github_api = GitHubModelsAPI()
            self.heart_tool = HeartDiseaseDBTool()
            self.cancer_tool = CancerDBTool()
            self.diabetes_tool = DiabetesDBTool()
            self.web_search_tool = MedicalWebSearchTool()
            self.api_available = True
        except Exception as e:
            print(f"Warning: GitHub API not available: {e}")
            self.api_available = False
            self.heart_tool = HeartDiseaseDBTool()
            self.cancer_tool = CancerDBTool()
            self.diabetes_tool = DiabetesDBTool()
            self.web_search_tool = MedicalWebSearchTool()
    
    def route_question(self, question):
        """Enhanced routing using GPT-4 mini for better understanding"""
        if self.api_available:
            return self._ai_powered_routing(question)
        else:
            return self._fallback_routing(question)
    
    def _ai_powered_routing(self, question):
        """Use GPT-4 mini to understand intent and route appropriately"""
        try:
            # First, determine the intent and routing
            routing_messages = [
                {
                    "role": "system",
                    "content": """You are a medical query router. Analyze the user's question and determine:

1. ROUTE_TYPE: Either "DATABASE" or "GENERAL"
   - DATABASE: For questions about statistics, data analysis, counts, averages, distributions from medical datasets
   - GENERAL: For questions about symptoms, treatments, definitions, causes, prevention

2. DATASET: If DATABASE route, specify "heart", "cancer", "diabetes", or "unknown"

3. SQL_INTENT: If DATABASE route, describe what SQL query would be needed

Respond ONLY in this JSON format:
{
  "route_type": "DATABASE" or "GENERAL",
  "dataset": "heart/cancer/diabetes/unknown",
  "sql_intent": "description of needed query",
  "reasoning": "brief explanation"
}"""
                },
                {
                    "role": "user",
                    "content": f"Route this medical question: {question}"
                }
            ]
            
            success, routing_response = self.github_api.generate_response(routing_messages, max_tokens=200)
            
            if success:
                try:
                    # Parse the routing decision
                    routing_data = json.loads(routing_response.strip())
                    route_type = routing_data.get("route_type", "").upper()
                    dataset = routing_data.get("dataset", "").lower()
                    sql_intent = routing_data.get("sql_intent", "")
                    
                    if route_type == "DATABASE":
                        return self._handle_database_query(question, dataset, sql_intent)
                    else:
                        return self._handle_general_query(question)
                        
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    return self._fallback_routing(question)
            else:
                return self._fallback_routing(question)
                
        except Exception as e:
            return f"Error in AI routing: {str(e)}"
    
    def _handle_database_query(self, question, dataset, sql_intent):
        """Handle database queries with AI-generated SQL"""
        try:
            # Select appropriate database tool
            if dataset == "heart":
                db_tool = self.heart_tool
                table_name = "heart"
                schema_info = "age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target"
            elif dataset == "cancer":
                db_tool = self.cancer_tool
                table_name = "thecancerdata1500v2"
                schema_info = "Age, Gender, BMI, Smoking, GeneticRisk, PhysicalActivity, AlcoholIntake, CancerHistory, Diagnosis"
            elif dataset == "diabetes":
                db_tool = self.diabetes_tool
                table_name = "diabetes"
                schema_info = "Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome"
            else:
                return "Please specify which dataset you're asking about (heart disease, cancer, or diabetes)."
            
            # Generate SQL query using AI
            sql_messages = [
                {
                    "role": "system",
                    "content": f"""You are a SQL query generator for medical data analysis. 

Table: {table_name}
Columns: {schema_info}

Generate a single, valid SQLite query to answer the user's question. 
- Use proper column names exactly as provided
- For outcomes: target=1 (heart disease), Diagnosis=1 (cancer), Outcome=1 (diabetes)
- Respond with ONLY the SQL query, no explanation or formatting"""
                },
                {
                    "role": "user",
                    "content": f"Generate SQL for: {question}"
                }
            ]
            
            success, sql_query = self.github_api.generate_response(sql_messages, max_tokens=150)
            
            if success:
                # Clean the SQL query
                sql_query = sql_query.strip().replace('```sql', '').replace('```', '').strip()
                
                # Execute the query
                query_success, result = db_tool.execute_query(sql_query)
                
                if query_success:
                    return self._format_database_response(result, question, sql_query)
                else:
                    # Fallback to basic query
                    return self._fallback_database_query(db_tool, question)
            else:
                return self._fallback_database_query(db_tool, question)
                
        except Exception as e:
            return f"Error processing database query: {str(e)}"
    
    def _handle_general_query(self, question):
        """Handle general medical questions using web search tool"""
        success, response = self.web_search_tool.search_medical_info(question)
        return response if success else "Unable to fetch medical information."
    
    def _format_database_response(self, result, question, sql_query):
        """Format database results using AI for natural language response"""
        try:
            columns = result["columns"]
            data = result["data"]
            
            # Use AI to format the response naturally
            format_messages = [
                {
                    "role": "system",
                    "content": """You are a medical data analyst. Format the SQL query results into a clear, natural language response. 
                    
Be concise but informative. If it's a single number, state it clearly. If it's multiple rows, summarize meaningfully.
Always provide medical context when relevant."""
                },
                {
                    "role": "user",
                    "content": f"""Question: {question}

SQL Query: {sql_query}

Results - Columns: {columns}
Data: {data[:10]}  

Format this into a natural language response."""
                }
            ]
            
            success, formatted_response = self.github_api.generate_response(format_messages, max_tokens=300)
            
            if success:
                return formatted_response
            else:
                # Fallback formatting
                if len(data) == 1 and len(columns) == 1:
                    return f"The result is: {data[0][0]}"
                else:
                    return f"Found {len(data)} results from the {sql_query.split('FROM')[1].split()[0]} dataset."
                    
        except Exception as e:
            return f"Results found but formatting error: {str(e)}"
    
    def _fallback_routing(self, question):
        """Fallback routing when AI is not available"""
        question_lower = question.lower()
        
        # Check if it's a data/statistics question
        data_keywords = ["count", "average", "mean", "distribution", "statistics", "data", "number", "percentage"]
        general_keywords = ["symptoms", "treatment", "cure", "what is", "definition", "causes"]
        
        is_data_question = any(keyword in question_lower for keyword in data_keywords)
        is_general_question = any(keyword in question_lower for keyword in general_keywords)
        
        if is_data_question:
            # Route to appropriate database
            if "heart" in question_lower:
                return self._fallback_database_query(self.heart_tool, question)
            elif "cancer" in question_lower:
                return self._fallback_database_query(self.cancer_tool, question)
            elif "diabetes" in question_lower:
                return self._fallback_database_query(self.diabetes_tool, question)
            else:
                return "Please specify which dataset you're asking about (heart, cancer, or diabetes)."
        
        elif is_general_question:
            # Route to web search
            success, result = self.web_search_tool.search_medical_info(question)
            return result if success else "Unable to fetch web information."
        
        else:
            return "Please ask about data statistics or general medical information (symptoms, treatments, etc.)."
    
    def _fallback_database_query(self, db_tool, question):
        """Fallback database query when AI is not available"""
        try:
            sql_query = db_tool.analyze_question(question)
            success, result = db_tool.execute_query(sql_query)
            
            if success:
                columns = result["columns"]
                data = result["data"]
                
                if len(data) == 1 and len(columns) == 1:
                    return f"The result is: {data[0][0]}"
                elif len(data) <= 10:
                    formatted_result = f"Query results:\n"
                    for row in data:
                        row_data = dict(zip(columns, row))
                        formatted_result += f"{row_data}\n"
                    return formatted_result
                else:
                    return f"Found {len(data)} results. First few: {data[:5]}"
            else:
                return f"Database error: {result}"
        except Exception as e:
            return f"Error processing question: {str(e)}"

# Initialize the AI agent
ai_agent = MedicalAIAgent()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'csvFile' not in request.files:
        return jsonify({"success": False, "message": "No file uploaded"})
    
    file = request.files['csvFile']
    dataset_type = request.form.get('datasetType')
    
    if file.filename == '':
        return jsonify({"success": False, "message": "No file selected"})
    
    if file and file.filename.endswith('.csv'):
        # Save uploaded file
        filename = f"{dataset_type}.csv"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Convert to SQLite
        success, message = CSVToSQLiteConverter.convert_csv_to_sqlite(filepath, dataset_type)
        
        return jsonify({"success": success, "message": message})
    
    return jsonify({"success": False, "message": "Invalid file format"})

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({"success": False, "message": "No question provided"})
    
    try:
        response = ai_agent.route_question(question)
        return jsonify({"success": True, "response": response})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error processing query: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
