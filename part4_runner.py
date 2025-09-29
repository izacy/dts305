from evaluation import DocumentBasedEvaluator


def main():    
    response = input("Ready to run document-based evaluation? (y/n): ").strip().lower()
    
    if response not in ['y', 'yes', ' ']:
        print("Evaluation cancelled")
        return
    
    try:
        evaluator = DocumentBasedEvaluator()
        success = evaluator.run_document_based_evaluation()

    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()