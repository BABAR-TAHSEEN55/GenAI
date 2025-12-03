import tiktoken


def Tokenization():
    text = "Hellow I'm Good"
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    response = enc.encode(text)
    print(response)


def main():
    # print("Hello from tokenization!")
    Tokenization()


if __name__ == "__main__":
    main()
