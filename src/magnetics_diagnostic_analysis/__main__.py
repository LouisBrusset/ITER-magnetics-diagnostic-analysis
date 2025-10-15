def ask_continuation(question="Continue workflow? (y/n): "):
    while (response := input(question).strip().lower()) not in ["y", "n", "yes", "no"]:
        print("Invalid answer. Please respond with 'y' (yes) or 'n' (no).")
    return response in ["y", "yes"]
