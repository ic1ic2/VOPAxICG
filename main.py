# main.py
from recommendation import run_recommendation

def main():
    user_text = """
    I feel stressed at work…
    my sleep is getting worse…
    """

    data, final_message = run_recommendation(user_text)

    print("\n--- RAW JSON ---")
    print(data)

    print("\n--- FINAL CHAT MESSAGE ---")
    print(final_message)


if __name__ == "__main__":
    main()
