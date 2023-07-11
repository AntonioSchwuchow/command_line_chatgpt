import os
import openai
from dotenv import load_dotenv
from colorama import Fore, Back, Style

# load values from the .env file if it exists
load_dotenv()

# configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

INSTRUCTIONS = """You are an AI assistant that is an expert in a car model called
"Volvo XC40" or "XC40" from Volvo, a car manufacturer.

Your name as an assistant is VolBot. Present yourself at the beginning of a conversation
with a greeting with your name.

You can provide specifications about the car, and general knowledge about the company.

If you are unable to provide an answer to a question, please respond with the phrase
"I´m just a simple chatBot for Volvo, please ask me anything about Volvo or the
XC40 model please."

Please aim to be as helpful as possible in all your responses.

Do not use any external URLs in you answers excepts the ones related to volvo.com

Format any lists on individual lines with a dash and a space in front of each item.

When giving specifications, ask for an area of interest before giving them. Choose 3 specifications related to what the user answers.

URLs that can be used in answers:
https://www.volvocars.com/us/cars/c40-electric/ for general information of the model

The next information is about the model, each dash (-) is about an area related to
the car as a product:

-Slogans or quick info about the car:

"Discover our first pure electric crossover with leather free
interior and Google built-in."

"A pure heart in a daring body"

"Express yourself. Contemporary design and a leather free interior set the tone
inside the C40 Recharge."

"More of what you want. Immerse yourself in smart features and enabling tech."

-Smart features:
Google built-in
Air purifier
Harman Kardon Premium Sound
360 camera
Cross traffic alert
Volvo Cars app

-Specifications
Electric range: 226 miles
Acceleration 4.5s (9-60mph)
Power 402 hp
Fast Charge 10-80% 37 minutes
Seats 5
Cargo Capacity 49 cu.ft.
Height 62.8 in
Width 75.2 in
Width including mirrors 80.1 in
Head room front/rear 39.4 in / 36.7 in
Length 174.8 in
Maximum towing capacity 2000 lbs

Twin Motor - Electric AWD
Automatic
402 hp
226 mi
Acceleration 4.5s (0-60 mph)
37 min Fast Charging
7-8 hours home charging
Charging equipment included: 11 kW onboard charger (Type 2).

-General information about the model
No compromises
The C40 Recharge doesn’t force you to choose between power and responsible driving.
Just take your seat, drive away and enjoy the comfort of one pedal drive and smooth
acceleration – with zero tailpipe emissions.

Responsible pleasure
Make every journey about much more than getting from A to B. In the C40 Recharge,
your commitment to a more sustainable future has never felt better."""

TEMPERATURE = 0.5
MAX_TOKENS = 500
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0.6
# limits how many questions we include in the prompt
MAX_CONTEXT_QUESTIONS = 10


def get_response(instructions, previous_questions_and_answers, new_question):
    """Get a response from ChatCompletion

    Args:
        instructions: The instructions for the chat bot - this determines how it will behave
        previous_questions_and_answers: Chat history
        new_question: The new question to ask the bot

    Returns:
        The response text
    """
    # build the messages
    messages = [
        { "role": "system", "content": instructions },
    ]
    # add the previous questions and answers
    for question, answer in previous_questions_and_answers[-MAX_CONTEXT_QUESTIONS:]:
        messages.append({ "role": "user", "content": question })
        messages.append({ "role": "assistant", "content": answer })
    # add the new question
    messages.append({ "role": "user", "content": new_question })

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    return completion.choices[0].message.content


def get_moderation(question):
    """
    Check the question is safe to ask the model

    Parameters:
        question (str): The question to check

    Returns a list of errors if the question is not safe, otherwise returns None
    """

    errors = {
        "hate": "Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.",
        "hate/threatening": "Hateful content that also includes violence or serious harm towards the targeted group.",
        "self-harm": "Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.",
        "sexual": "Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).",
        "sexual/minors": "Sexual content that includes an individual who is under 18 years old.",
        "violence": "Content that promotes or glorifies violence or celebrates the suffering or humiliation of others.",
        "violence/graphic": "Violent content that depicts death, violence, or serious physical injury in extreme graphic detail.",
    }
    response = openai.Moderation.create(input=question)
    if response.results[0].flagged:
        # get the categories that are flagged and generate a message
        result = [
            error
            for category, error in errors.items()
            if response.results[0].categories[category]
        ]
        return result
    return None


def main():
    os.system("cls" if os.name == "nt" else "clear")
    # keep track of previous questions and answers
    previous_questions_and_answers = []
    while True:
        # ask the user for their question
        new_question = input(
            Fore.GREEN + Style.BRIGHT + "What can I get you?: " + Style.RESET_ALL
        )
        # check the question is safe
        errors = get_moderation(new_question)
        if errors:
            print(
                Fore.RED
                + Style.BRIGHT
                + "Sorry, you're question didn't pass the moderation check:"
            )
            for error in errors:
                print(error)
            print(Style.RESET_ALL)
            continue
        response = get_response(INSTRUCTIONS, previous_questions_and_answers, new_question)

        # add the new question and answer to the list of previous questions and answers
        previous_questions_and_answers.append((new_question, response))

        # print the response
        print(Fore.CYAN + Style.BRIGHT + "Here you go: " + Style.NORMAL + response)


if __name__ == "__main__":
    main()
