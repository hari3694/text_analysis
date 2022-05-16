from transformers import pipeline, AutoTokenizer, AutoModelWithLMHead


text = '''Amazon.com, Inc. is an American multinational technology company which focuses on e-commerce, cloud computing, digital streaming, and artificial intelligence. It has been referred to as "one of the most influential economic and cultural forces in the world", and is one of the world's most valuable brands. It is one of the Big Five American information technology companies, alongside Alphabet, Apple, Meta, and Microsoft.

Amazon was founded by Jeff Bezos from his garage in Bellevue, Washington, on July 5, 1994. Initially an online marketplace for books, it has expanded into a multitude of product categories: a strategy that has earned it the moniker The Everything Store. It has multiple subsidiaries including Amazon Web Services (cloud computing), Zoox (autonomous vehicles), Kuiper Systems (satellite Internet), Amazon Lab126 (computer hardware R&D). Its other subsidiaries include Ring, Twitch, IMDb, and Whole Foods Market. Its acquisition of Whole Foods in August 2017 for US$13.4 billion substantially increased its footprint as a physical retailer.

Amazon has earned a reputation as a disruptor of well-established industries through technological innovation and mass scale. As of 2021, it is the world's largest Internet company, online marketplace, AI assistant provider, cloud computing platform, and live-streaming service as measured by revenue and market share. In 2021, it surpassed Walmart as the world's largest retailer outside of China, driven in large part by its paid subscription plan, Amazon Prime, which has over 200 million subscribers worldwide. It is the second-largest private employer in the United States.

Amazon also distributes a variety of downloadable and streaming content through its Amazon Prime Video, Amazon Music, Twitch, and Audible units. It publishes books through its publishing arm, Amazon Publishing, film and television content through Amazon Studios, and is the owner of film and television studio Metro-Goldwyn-Mayer since 2022. It also produces consumer electronics—most notably, Kindle e-readers, Echo devices, Fire tablets, and Fire TV.

Amazon has been criticized for practices including technological surveillance overreach, a hyper-competitive and demanding work culture, tax avoidance, and anti-competitive behavior.
'''

text1 = '''I gave the Hunger Gamrs 5 stars because I simply could not put it down! The characters are riveting and the book is set in a dystopian world called Panem which hosts the annual Hunger Games, a tournament where two tributes from each of the twelve districts fight to death until only one victor remains. If you like hard science fiction the you'll love this book!'''

def summarise_text(text):
    classifier = pipeline("summarization")
    summary = classifier(text, max_length=50)
    print(summary)

def sentiment_analysis_text(text):
    classifier = pipeline("sentiment-analysis")
    result = classifier(text)
    print(result)


def get_intent(event, max_length=16):
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-e2m-intent")
    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-e2m-intent")
    input_text = "%s </s>" % event
    features = tokenizer([input_text], return_tensors='pt')

    output = model.generate(input_ids=features['input_ids'],
           attention_mask=features['attention_mask'],
           max_length=max_length)

    return tokenizer.decode(output[0])

event = "PersonX takes PersonY home"
get_intent(event)

if __name__ == "__main__":
    summarise_text(text=text)
    # sentiment_analysis_text("I am angry at you")
