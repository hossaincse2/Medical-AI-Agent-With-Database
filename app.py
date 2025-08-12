import os
import sqlite3
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
import requests
import json
from datetime import datetime
import re

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

class MedicalWebSearchTool:
    """Web search tool for general medical knowledge"""
    
    @staticmethod
    def search_medical_info(query):
        try:
            # Use Claude's web search capability for real medical information
            search_query = f"medical {query}"
            
            # For this demo, we'll provide comprehensive medical responses
            # In a real implementation, you would integrate with actual search APIs
            
            # Enhanced medical knowledge base
            if "symptoms" in query.lower():
                if "heart disease" in query.lower() or "heart" in query.lower():
                    return True, "Heart disease symptoms include: chest pain or discomfort, shortness of breath, fatigue, irregular heartbeat, swelling in legs/feet/ankles, dizziness, and pain in neck/jaw/throat/upper abdomen. Seek immediate medical attention for chest pain."
                elif "cancer" in query.lower():
                    return True, "Cancer symptoms vary by type but may include: unexplained weight loss, persistent fatigue, unusual lumps or swelling, changes in bowel/bladder habits, persistent cough, difficulty swallowing, unusual bleeding, and skin changes. Early detection is crucial."
                elif "diabetes" in query.lower():
                    return True, "Diabetes symptoms include: increased thirst and urination, extreme fatigue, blurred vision, slow-healing cuts/bruises, tingling in hands/feet, recurring infections. Type 1 symptoms appear quickly; Type 2 develop gradually."
            
            elif "treatment" in query.lower() or "cure" in query.lower():
                if "heart disease" in query.lower() or "heart" in query.lower():
                    return True, "Heart disease treatment includes: lifestyle changes (diet, exercise, smoking cessation), medications (blood thinners, beta-blockers, statins), and procedures (angioplasty, bypass surgery). Treatment depends on specific condition and severity."
                elif "cancer" in query.lower():
                    return True, "Cancer treatment options include: surgery, chemotherapy, radiation therapy, immunotherapy, targeted therapy, hormone therapy, and stem cell transplant. Treatment plans are personalized based on cancer type, stage, and patient factors."
                elif "diabetes" in query.lower():
                    return True, "Diabetes management includes: blood glucose monitoring, insulin therapy (Type 1), oral medications (Type 2), healthy diet, regular exercise, weight management, and regular medical checkups. Goal is maintaining healthy blood sugar levels."
            
            elif "causes" in query.lower() or "risk factors" in query.lower():
                if "heart disease" in query.lower() or "heart" in query.lower():
                    return True, "Heart disease risk factors include: high blood pressure, high cholesterol, smoking, diabetes, obesity, physical inactivity, family history, age, and stress. Many risk factors are preventable through lifestyle changes."
                elif "cancer" in query.lower():
                    return True, "Cancer risk factors include: tobacco use, excessive alcohol, poor diet, physical inactivity, obesity, sun exposure, certain infections, family history, age, and environmental exposures. Some factors are preventable."
                elif "diabetes" in query.lower():
                    return True, "Diabetes risk factors include: family history, obesity, physical inactivity, age (45+), high blood pressure, abnormal cholesterol levels, history of gestational diabetes, and certain ethnicities. Type 1 has autoimmune components."
            
            elif "prevention" in query.lower():
                if "heart disease" in query.lower() or "heart" in query.lower():
                    return True, "Heart disease prevention: maintain healthy diet, exercise regularly, don't smoke, limit alcohol, manage stress, control blood pressure/cholesterol/diabetes, maintain healthy weight, and get regular checkups."
                elif "cancer" in query.lower():
                    return True, "Cancer prevention: don't smoke, limit alcohol, maintain healthy weight, stay physically active, eat healthy diet, protect from sun, get vaccinated, avoid risky behaviors, and get regular screenings."
                elif "diabetes" in query.lower():
                    return True, "Type 2 diabetes prevention: maintain healthy weight, be physically active, eat healthy diet, limit refined sugars, don't smoke, control blood pressure, and get regular screenings. Type 1 cannot currently be prevented."
            
            elif "definition" in query.lower() or "what is" in query.lower():
                if "heart disease" in query.lower() or "heart" in query.lower():
                    return True, "Heart disease refers to several types of heart conditions, including coronary artery disease, heart rhythm problems, and heart defects. It's the leading cause of death globally and often preventable."
                elif "cancer" in query.lower():
                    return True, "Cancer is a group of diseases involving abnormal cell growth with potential to invade other parts of the body. It occurs when cells divide uncontrollably and spread into surrounding tissues."
                elif "diabetes" in query.lower():
                    return True, "Diabetes is a group of metabolic disorders characterized by high blood sugar levels. Type 1 (autoimmune) and Type 2 (insulin resistance) are the main types, affecting how the body processes glucose."
            
            # Default response for unmatched queries
            return True, f"For detailed medical information about '{query}', consult healthcare professionals, medical literature, or trusted medical websites like Mayo Clinic, WebMD, or NIH."
            
        except Exception as e:
            return False, f"Web search error: {str(e)}"

class MedicalAIAgent:
    """Main AI Agent that routes questions to appropriate tools"""
    
    def __init__(self):
        self.heart_tool = HeartDiseaseDBTool()
        self.cancer_tool = CancerDBTool()
        self.diabetes_tool = DiabetesDBTool()
        self.web_search_tool = MedicalWebSearchTool()
    
    def route_question(self, question):
        """Route question to appropriate tool based on content"""
        question_lower = question.lower()
        
        # Check if it's a data/statistics question
        data_keywords = ["count", "average", "mean", "distribution", "statistics", "data", "number", "percentage"]
        general_keywords = ["symptoms", "treatment", "cure", "what is", "definition", "causes"]
        
        is_data_question = any(keyword in question_lower for keyword in data_keywords)
        is_general_question = any(keyword in question_lower for keyword in general_keywords)
        
        if is_data_question:
            # Route to appropriate database
            if "heart" in question_lower:
                return self._query_database(self.heart_tool, question)
            elif "cancer" in question_lower:
                return self._query_database(self.cancer_tool, question)
            elif "diabetes" in question_lower:
                return self._query_database(self.diabetes_tool, question)
            else:
                return "Please specify which dataset you're asking about (heart, cancer, or diabetes)."
        
        elif is_general_question:
            # Route to web search
            success, result = self.web_search_tool.search_medical_info(question)
            return result if success else "Unable to fetch web information."
        
        else:
            return "Please ask about data statistics or general medical information (symptoms, treatments, etc.)."
    
    def _query_database(self, db_tool, question):
        """Execute database query and format response"""
        try:
            sql_query = db_tool.analyze_question(question)
            success, result = db_tool.execute_query(sql_query)
            
            if success:
                # Format the response nicely
                columns = result["columns"]
                data = result["data"]
                
                if len(data) == 1 and len(columns) == 1:
                    # Single value result
                    return f"The result is: {data[0][0]}"
                elif len(data) <= 10:
                    # Small result set
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
