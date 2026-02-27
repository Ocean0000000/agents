from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastmcp import FastMCP

class Student(BaseModel):
    id: int
    name: str
    gpa: float

# my api
app = FastAPI()

# my database
students: dict[int, Student] = {}

@app.get("/students", response_model=list[Student])
def get_students():
    return list(students.values())

@app.get("/students/{student_id}", response_model=Student)
def get_student(student_id: int):
    if student_id in students:
        return students[student_id]
    raise HTTPException(status_code=404, detail="Student not found")

@app.post("/students", response_model=Student)
def create_student(student: Student):
    if student.id not in students:
        students[student.id] = student
        print(f"Created student: {student}")
        return student    
    raise HTTPException(status_code=400, detail="Student with this ID already exists")

@app.put("/students/{student_id}", response_model=Student)
def update_student(student_id: int, student: Student):
    if student_id in students:
        students[student_id] = student
        return student
    raise HTTPException(status_code=404, detail="Student not found")

@app.delete("/students/{student_id}", response_model=dict[str, str])
def delete_student(student_id: int):
    if student_id in students:
        del students[student_id]
        return {"message": "Student deleted"}
    raise HTTPException(status_code=404, detail="Student not found")

mcp = FastMCP.from_fastapi(app, name="Student Information System")
app.mount("/mcp", mcp.http_app())

if __name__ == "__main__":
    mcp.run(transport="streamable-http")