import emoji, re

def extract_emojis(iggy):
    return re.sub(r'([^0-9A-Za-z \t])|(@[A-Za-z0-9]+)|(http\S+)', '', iggy), ' '.join(c for c in iggy if c in emoji.UNICODE_EMOJI)
def subd(text):
    return re.sub(r'( +)', ' ', text)

string = extract_emojis("🤔 🙈 me así, bla es se 😌 ds 💕👭👙 Data: https://t.co/VI927y5OZd https://t.co/YqgFggXuuGRT @UserExperienceU: Paradigm Interactions R&amp;D company UbiNET issues two cryptocurrencies the #ThingCoin and #ThingCoin #ICO #Crypto #IoT #…Buying #Bitcoin is like buying air 🤣 @PeterSchiff https://t.co/tzIahQUMbA RT @T21094: 仮想通貨 BDA☺️ jbjhc")
print(type(string))
text = subd(string[0])
boop = string[1]
#string = ' '.join(string)
text = text + ' ' + boop
print(text)
#print(text)