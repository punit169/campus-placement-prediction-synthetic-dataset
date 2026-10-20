"""
Campus Placement Benchmark Dataset Generator
Course: IT462 Machine Learning Project
Description: Synthesizes a high-fidelity, large-scale (100,000 records) campus placement dataset
             incorporating academic, technical, project, and internship indicators.
"""

import numpy as np
import pandas as pd
import os

def generate_placement_dataset(n_samples=100000, random_state=42):
    np.random.seed(random_state)
    
    # 1. Student Identification
    student_ids = [f"STU_{i+1:06d}" for i in range(n_samples)]
    
    # 2. Academic Factors
    cpi = np.random.normal(loc=7.3, scale=1.1, size=n_samples)
    cpi = np.clip(np.round(cpi, 2), 4.5, 10.0)
    
    # Branch distribution (Tech / Engineering campus)
    branches = ['CSE', 'ICT', 'IT', 'ECE', 'MnC', 'EE', 'Mechanical', 'Civil']
    branch_probs = [0.25, 0.20, 0.15, 0.15, 0.08, 0.07, 0.05, 0.05]
    branch = np.random.choice(branches, size=n_samples, p=branch_probs)
    
    # Attendance Rate (%): 60% to 100%
    attendance = np.clip(np.round(np.random.normal(loc=82.5, scale=8.5, size=n_samples), 1), 60.0, 100.0)
    
    # Backlogs (failed/uncleared courses)
    backlog_probs = [0.72, 0.16, 0.07, 0.03, 0.02]
    backlogs = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=backlog_probs)
    cpi = np.where(backlogs >= 2, np.clip(cpi - np.random.uniform(0.3, 0.7, size=n_samples), 4.5, 10.0), cpi)
    cpi = np.round(cpi, 2)
    
    # 3. Technical & Coding Skills
    base_coding = (cpi / 10.0) * 5.0 + np.random.normal(loc=2.2, scale=1.8, size=n_samples)
    coding_score = np.clip(np.round(base_coding, 1), 1.0, 10.0)
    
    # DSA (Data Structures & Algorithms) score (1.0 to 10.0)
    dsa_score = np.clip(np.round(coding_score * 0.7 + np.random.normal(loc=1.6, scale=1.4, size=n_samples), 1), 1.0, 10.0)
    
    # DSA / Competitive Programming Problems Solved (0 to 800)
    dsa_problems_solved = np.clip(np.round(dsa_score * 42 + np.random.exponential(scale=35, size=n_samples)), 0, 800).astype(int)
    
    # 4. Project Portfolio
    projects_count = np.clip(np.round(np.random.poisson(lam=2.4, size=n_samples)), 0, 6).astype(int)
    major_project_prob = np.clip(0.12 + 0.13 * projects_count + 0.05 * (coding_score - 5.0), 0.05, 0.95)
    has_major_project = (np.random.rand(n_samples) < major_project_prob).astype(int)
    
    # 5. Work Experience & Internships
    internship_prob = np.clip(0.12 + 0.09 * (cpi - 6.5) + 0.08 * (coding_score - 5.0), 0.04, 0.85)
    internships = np.random.binomial(n=3, p=np.clip(internship_prob / 3.0, 0.02, 0.65), size=n_samples)
    
    duration_per_internship = np.random.choice([2, 3, 6], size=n_samples, p=[0.45, 0.40, 0.15])
    experience_months = internships * duration_per_internship
    
    # 6. Aptitude, Soft Skills & Extracurriculars
    aptitude_score = np.clip(np.round(np.random.normal(loc=71.0, scale=13.5, size=n_samples), 1), 25.0, 100.0)
    communication_skills = np.clip(np.round(np.random.normal(loc=6.8, scale=1.5, size=n_samples), 1), 2.0, 10.0)
    hackathons_attended = np.clip(np.random.poisson(lam=1.1, size=n_samples), 0, 5).astype(int)
    certifications_count = np.clip(np.random.poisson(lam=1.4, size=n_samples), 0, 5).astype(int)
    
    # 7. Placement Status (Logistic Decision Function calibrated to ~64.5% placement rate)
    z = (
        1.25 * (cpi - 7.0)
        + 0.55 * (coding_score - 5.5)
        + 0.45 * (dsa_score - 5.5)
        + 0.40 * (projects_count - 2.0)
        + 0.50 * has_major_project
        + 0.75 * (internships - 0.8)
        + 0.03 * (aptitude_score - 70.0)
        + 0.35 * (communication_skills - 6.5)
        - 1.60 * backlogs
        + np.random.normal(loc=0.0, scale=1.2, size=n_samples)
        + 1.45
    )
    
    placement_prob = 1.0 / (1.0 + np.exp(-z))
    placement_status = (np.random.rand(n_samples) < placement_prob).astype(int)
    
    # 8. Salary Package (LPA - Lakhs Per Annum) for Placed Candidates
    base_lpa = 4.2 + 0.75 * (cpi - 6.0) + 1.0 * (coding_score - 4.5) + 0.8 * (dsa_score - 4.5) + 1.4 * internships + 0.5 * projects_count
    elite_bonus = np.where((coding_score >= 8.2) & (dsa_score >= 8.0), np.random.exponential(scale=7.5, size=n_samples), 0.0)
    calculated_salary = np.round(np.clip(base_lpa + elite_bonus + np.random.normal(0, 1.1, size=n_samples), 3.5, 48.0), 2)
    salary_lpa = np.where(placement_status == 1, calculated_salary, np.nan)
    
    # Construct DataFrame
    df = pd.DataFrame({
        'student_id': student_ids,
        'branch': branch,
        'cpi': cpi,
        'attendance_rate': attendance,
        'backlogs': backlogs,
        'coding_score': coding_score,
        'dsa_score': dsa_score,
        'dsa_problems_solved': dsa_problems_solved,
        'projects_count': projects_count,
        'has_major_project': has_major_project,
        'internships': internships,
        'experience_months': experience_months,
        'aptitude_score': aptitude_score,
        'communication_skills': communication_skills,
        'hackathons_attended': hackathons_attended,
        'certifications_count': certifications_count,
        'placement_status': placement_status,
        'salary_lpa': salary_lpa
    })
    
    return df

if __name__ == '__main__':
    print("Generating 100,000-row College Placement dataset...")
    df = generate_placement_dataset(100000)
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "student_placement_data.csv")
    
    df.to_csv(out_path, index=False)
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Dataset generated successfully at: {out_path}")
    print(f"Shape: {df.shape}")
    print(f"File Size: {file_size_mb:.2f} MB")
    print(f"Placement Rate: {df['placement_status'].mean()*100:.2f}%")
