import warnings
warnings.filterwarnings('ignore')

import yaml
from crewai import Agent, Task, Crew, LLM
import streamlit as st
import pandas as pd

#Set AI model
from dotenv import load_dotenv
import os
load_dotenv()

llm = LLM(
                model="llama-3.1-8b-instant",
                temperature=0.7,
                base_url="https://api.groq.com/openai/v1",
                api_key = os.getenv('API_KEY')
            )


#Loading Agents and Task Yaml files
##Define file path for yaml configurations
files = {
        'agents': 'agents.yaml',
        'tasks': 'tasks.yaml'
    }

##Load configurations from Yaml files
configs ={}
for config_type, file_path in files.items():
    with open(file_path, 'r') as file:
        configs[config_type] = yaml.safe_load(file)

##Assign loaded configurations to specific variables
agents_config = configs['agents']
tasks_config = configs['tasks']

#Creating pydantic models for structured outputs
from typing import List
from pydantic import BaseModel, Field

class TaskEstimate(BaseModel):
    task_name: str = Field(..., description="Name of the Task")
    estimated_time_hours: float = Field(..., description="Estimated time to complete the task in hours")
    required_resources: List[str] = Field(..., description="List of resources required to complete the task")

class Milestone(BaseModel):
    milestone_name: str = Field(..., description="Name of the milestone")
    tasks: List[str] = Field(..., description="List of task IDs associated with this milestone")

class ProjectPlan(BaseModel):
    tasks: List[TaskEstimate] = Field(..., description="List of tasks with their estimates")
    milestones: List[Milestone] = Field(..., description="List of project milestones")

##Agents, Tasks, Crew

#Creating Agents: Project Planner, Project Estimator, Resources Allocator
project_planning_agent = Agent(
    config=agents_config['project_planning_agent'],
    llm=llm
)

estimation_agent = Agent(
    config=agents_config['estimation_agent'],
    llm=llm
)

resource_allocation_agent = Agent(
    config=agents_config['resource_allocation_agent'],
    llm=llm
)

#Creating Tasks: Task Breakdown, Task Time Estimation, Resource Allocation
task_breakdown = Task(
    config=tasks_config['task_breakdown'],
    agent=project_planning_agent
)

time_resource_estimation = Task(
    config=tasks_config['time_resource_estimation'],
    agent=estimation_agent
)

resource_allocation = Task(
    config=tasks_config['resource_allocation'],
    agent=resource_allocation_agent,
    output_pydantic=ProjectPlan
)

#Creating the Crew
crew = Crew(
    agents=[
        project_planning_agent,
        estimation_agent,
        resource_allocation_agent
    ],
    tasks=[
        task_breakdown,
        time_resource_estimation,
        resource_allocation
    ],
    verbose=True
)

def project_planning(input_data):
    result = str(crew.kickoff(
            inputs=input_data
    ))
    return result


def running_costs():
    costs = 0.150 * (crew.usage_metrics.prompt_tokens + crew.usage_metrics.completion_tokens) / 1_000_000
    print(f"Total costs: ${costs:.4f}")
    df_usage_metrics = pd.DataFrame([crew.usage_metrics.model_dump()])
    return df_usage_metrics

def logs_task (input_data):
    result = crew.kickoff(
        inputs=input_data
    )
    result.pydantic.dict()
    tasks = result.pydantic.dict()['tasks']
    df_tasks = pd.DataFrame(tasks)
    return df_tasks.style.set_table_attributes('border="1').set_caption("Task Details").set_table([{'selector': 'th, td', 'props': [('font-size', '120%')]}])

def logs_milestones (input_data):
    result = crew.kickoff(
        inputs=input_data
    )
    result.pydantic.dict()
    milestones = result.pydantic.dict()['milestones']
    df_milestones = pd.DataFrame(milestones)
    return df_milestones


def main():
    # Giving a title
    st.title('AI Project Planner By Steeve')
    col1, col2 = st.columns(2)


    # Getting the input data from the user
    with col1:
        project_type = st.text_input('Project: (Ex: Website)')
        industry = st.text_input('Industry: (Ex: Technology)')
        project_objectives = st.text_area(f'Project Objectives: (Ex:Create a website for my small business)', height=300)


    with col2:
        team_members = st.text_area(f'List of Team Members Names (and Roles):', height=300)

        #project_requirements_text_area = st.text_area('List of Project Requirements:', height=300)
        project_requirements = st.text_area(f'List of Project Requirements:', height=300)


    inputs = [project_type, industry, project_objectives, team_members, project_requirements]
    inputs_dict = {'project_type': inputs[0], 'industry': inputs[1], 'project_objectives': inputs[2], 'team_members': inputs[3], 'project_requirements': inputs[4]}

    # code for outputs

    plan = ''

    # creating a button for the outputs
    if st.button("Create a plan"):
        plan = project_planning(inputs_dict)
        if isinstance(plan, dict):
            for key, value in plan.items():
                st.write(f"{key}: {value}")
        elif isinstance(plan, list):
            for item in plan:
                st.write(item)
        else:
            st.write(plan)





if __name__ == '__main__':
    main()