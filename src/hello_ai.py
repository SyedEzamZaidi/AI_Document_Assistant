def greet_ai_engineer(name):
    message = f"Hello {name} ! Welcome to AI Engineering"
    return message

def show_project_info():
    project_name = "Intelligent Document Assistant"
    version = "1.0.0"
    features = ["Multi-model RAG system","Automated quality evaluation","Cost optimization tracking","Power BI dashboard integration"]

    print("="*50)
    print(f"Project is {project_name}")
    print(f"Version is {version}")
    print("="*50)
    print("\n Planned Features:")

    for i, feature in enumerate(features,1):
        print(f"{i}.{feature}")

    print("="*50)

if __name__ == "__main__":
    greeting = greet_ai_engineer("Ezam")
    print(greeting)
    print()
    show_project_info()

    print("\n Python environment is ready!")
    print(" VS Code is configured!")
    print(" Virtual environment is active!")
    print("\n Next step: Building the RAG system!\n")






