from modal import App, Secret, is_local

custom_secret = Secret.from_name("custom-secret")

if is_local():
    from dotenv import load_dotenv
    load_dotenv()

app = App(name="new-munshi", secrets=[custom_secret])
