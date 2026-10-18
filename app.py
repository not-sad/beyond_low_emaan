# Interactive pastoral counselor client script
from generation import load_model, generate_response, SYSTEM_B

def main():
    print("Loading Pastoral Counseling Assistant...")
    tok, mod = load_model()
    print("Assistant active. Type 'exit' to quit.\n")
    while True:
        user_msg = input("User: ")
        if user_msg.lower() == 'exit':
            break
        resp = generate_response(tok, mod, user_msg, system_prompt=SYSTEM_B)
        print(f"\nAssistant: {resp}\n" + "-"*40)

if __name__ == '__main__':
    main()
