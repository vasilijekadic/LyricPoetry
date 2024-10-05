from openai import OpenAI
import pandas as pd
import seaborn as sb
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix

client = OpenAI(
    api_key='api key'
)

LABELS = ["Ljubavna poezija", "Misaono - refleksivna poezija", "Rodoljubiva poezija"]
LABELS_ENGLISH = ["Love poem", "Thought-reflexive poem", "Patriotic poem"]


def translate_labels(text):
    dict = {'Love poem': 'Ljubavna poezija', 'Thought-reflexive poem': 'Misaono - refleksivna poezija',
            'Patriotic poem': 'Rodoljubiva poezija'}
    return dict[text]


def classify_text(text, labels):
    # prompt = (f"This is a poem classification task. The poem is written in Serbian."
    #           f"Classify the following poem into the appropriate category from the list: {labels}."
    #           f"Your response should only be the assigned label. The label must be one of the labels from the list: {labels}."
    #           f"Here are some examples:\n"
    #           f"Poem: Prelepa reko! dok tvoj pramen kristalno čist treperi, ti žarke si lepote znamen – srce neskrito ljupko – smelih draži klupko u starog Alberta kćeri; Kad ona tvom se valu smeši – što trepti kroz odsjaje – tad od svih potoka lepši njen poklonik postaje; jer, kao tvoj val, on sred srca lik čuva njen duboko – i drhti dok se nad njim zrca duše joj željno oko.\n Label: Love poem\n\n"
    #           f"Poem: Kao kad zaroniš do dna mora, A samo kamenčić zgrneš u dlan, Tako ti počne neka zora, Tako se završi po neki dan. I suze detinje kaplju sve teže Na snove prazne k’o prazne mreže. A nekad onako, kao od šale, Padne pred tebe zvezda prava, A ti je šutneš vrhom sandale I odeš dalje. I kad se spava, Kad nebo zaljulja sva svoj klatna, Čitava noć ti od snova – zlatna. Posle detinjstva šta se sve menja? Opet se ređaju snovi i snovi. Između zvezda i kamenja Jastuk kroz život i dalje plovi. Samo je nešto teža glava. Manje se sanja – više se spava.  \n Label: Thought-reflexive poem\n\n"
    #           f"Poem: Na mekom, toplom krilu Jedinka sina njiha, Ljubi ga noć i dan; Cvećem mu vlasi kiti, Pesmom mu sladi san. Raste joj sinak, raste, Na svoje noge staje, Vasceo majčin svet – Oh, nije šala, nije: Na grani jedan cvet. Raste joj sinak, raste, A majka dršće, strepi, U nežnom srcu svom, Da ne bi na cvet pao Iz vedra neba grom. Raste joj sinak, raste, Jedino blago majci Koje joj dade Bog. Ko ne bi brižno čuvo „Zenicu oka svog?“ Raste joj sinak, raste, Majka bi zvezde s neba Skidala svaki čas, Da sinku, svom jedinku, Od zvezda splete pas. Sinak se snagom paše, A majka sneva svate, Veselja nada svog; Topi se od milina Kraj sina jedinog. Al’ začu s’ bojna truba: „Za Srpstvo, za slobodu!“ – Majka mu paše mač. A kad je pao, niko Nije joj čuo plač.  \n Label: Patriotic poem\n\n"
    #           f"Poem: Ko sam? Šta sam? Ja sam samo sanjar, Čiji pogled gasne u magli i memli, Živio sam usput, ko da sanjam, Kao mnogi drugi ljudi na toj zemlji. I tebe sad ljubim po navici ,dete, Zato što sam mnoge ljubio, bolećiv, Zato usput, ko što palim cigarete, Govorim i šapćem zaljubljene reči. „Draga moja“,“mila“,“znaj, doveka“ A u duši vazda ista pustoš zrači; Ako dirneš strast u čovekovu biću Istine, bez sumnje, nikad nećeš naći. Zato moja duša ne zna što je jeza Odbijenih želja, neshvaćene tuge. Ti si, moja gipka, lakonoga breza, Stvorena za mene i za mnoge druge. Ali ako tražeć neku srodnu dušu, Vezan protiv želje, utonem u seti, Nikad necu da te ljubomorom gušim, Nikad necu tebe grditi ni kleti. Ko sam? Šta sam? Ja sam samo sanjar, Čiji pogled gasne u magli i memli, I volim te usput, ko da sanjam, Kao mnoge druge na toj zemlji.\n Label: Love poem\n\n"
    #           f"Poem: Tiha noćca nastupila, Poče blago širiti se, Nebo plavo, bez oblačka, Stade zv’jezdam’ kititi se. Gledao sam kako milo Svaka svoje zrake daje, Al’ sam gled’o najradije Jato od njih, što tu sjaje. Uporedo sve su stale, Vođa im je prva bila, Pa su nebom putovale Šireć’ svoja zlatna krila. Tad pomislih: mili Bože, Srbadija kad će tako, Poći skupa? A odgovor: – Kad neslogu sprži pak’o… \n Label: Patriotic poem\n\n"
    #           f"Poem: {text}\nLabel: ")

    prompt = (f"Ovo je problem klasifikacije pesama napisanih na srpskom jeziku."
              f"Sledećoj pesmi dodeli jednu od kategorija iz sledeće liste: {labels}."
              f"Tvoj odgovor treba da bude samo dodeljena labela. Labela mora biti jedna iz sledeće liste: {labels}."
              f"Evo nekoliko primera:\n"
              f"Pesma: Prelepa reko! dok tvoj pramen kristalno čist treperi, ti žarke si lepote znamen – srce neskrito ljupko – smelih draži klupko u starog Alberta kćeri; Kad ona tvom se valu smeši – što trepti kroz odsjaje – tad od svih potoka lepši njen poklonik postaje; jer, kao tvoj val, on sred srca lik čuva njen duboko – i drhti dok se nad njim zrca duše joj željno oko.\n Labela: Ljubavna poezija\n\n"
              f"Pesma: Kao kad zaroniš do dna mora, A samo kamenčić zgrneš u dlan, Tako ti počne neka zora, Tako se završi po neki dan. I suze detinje kaplju sve teže Na snove prazne k’o prazne mreže. A nekad onako, kao od šale, Padne pred tebe zvezda prava, A ti je šutneš vrhom sandale I odeš dalje. I kad se spava, Kad nebo zaljulja sva svoj klatna, Čitava noć ti od snova – zlatna. Posle detinjstva šta se sve menja? Opet se ređaju snovi i snovi. Između zvezda i kamenja Jastuk kroz život i dalje plovi. Samo je nešto teža glava. Manje se sanja – više se spava.  \n Labela: Misaono - refleksivna poezija\n\n"
              f"Pesma: Na mekom, toplom krilu Jedinka sina njiha, Ljubi ga noć i dan; Cvećem mu vlasi kiti, Pesmom mu sladi san. Raste joj sinak, raste, Na svoje noge staje, Vasceo majčin svet – Oh, nije šala, nije: Na grani jedan cvet. Raste joj sinak, raste, A majka dršće, strepi, U nežnom srcu svom, Da ne bi na cvet pao Iz vedra neba grom. Raste joj sinak, raste, Jedino blago majci Koje joj dade Bog. Ko ne bi brižno čuvo „Zenicu oka svog?“ Raste joj sinak, raste, Majka bi zvezde s neba Skidala svaki čas, Da sinku, svom jedinku, Od zvezda splete pas. Sinak se snagom paše, A majka sneva svate, Veselja nada svog; Topi se od milina Kraj sina jedinog. Al’ začu s’ bojna truba: „Za Srpstvo, za slobodu!“ – Majka mu paše mač. A kad je pao, niko Nije joj čuo plač.  \n Labela: Rodoljubiva poezija\n\n"
              f"Pesma: Ko sam? Šta sam? Ja sam samo sanjar, Čiji pogled gasne u magli i memli, Živio sam usput, ko da sanjam, Kao mnogi drugi ljudi na toj zemlji. I tebe sad ljubim po navici ,dete, Zato što sam mnoge ljubio, bolećiv, Zato usput, ko što palim cigarete, Govorim i šapćem zaljubljene reči. „Draga moja“,“mila“,“znaj, doveka“ A u duši vazda ista pustoš zrači; Ako dirneš strast u čovekovu biću Istine, bez sumnje, nikad nećeš naći. Zato moja duša ne zna što je jeza Odbijenih želja, neshvaćene tuge. Ti si, moja gipka, lakonoga breza, Stvorena za mene i za mnoge druge. Ali ako tražeć neku srodnu dušu, Vezan protiv želje, utonem u seti, Nikad necu da te ljubomorom gušim, Nikad necu tebe grditi ni kleti. Ko sam? Šta sam? Ja sam samo sanjar, Čiji pogled gasne u magli i memli, I volim te usput, ko da sanjam, Kao mnoge druge na toj zemlji.\n Labela: Ljubavna poezija\n\n"
              f"Pesma: Tiha noćca nastupila, Poče blago širiti se, Nebo plavo, bez oblačka, Stade zv’jezdam’ kititi se. Gledao sam kako milo Svaka svoje zrake daje, Al’ sam gled’o najradije Jato od njih, što tu sjaje. Uporedo sve su stale, Vođa im je prva bila, Pa su nebom putovale Šireć’ svoja zlatna krila. Tad pomislih: mili Bože, Srbadija kad će tako, Poći skupa? A odgovor: – Kad neslogu sprži pak’o… \n Labela: Rodoljubiva poezija\n\n"
              f"Pesma: {text}\nLabela: ")

    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="gpt-3.5-turbo",
        max_tokens=100,
        temperature=0
    )

    labels = response.choices[0].message.content.strip()
    return labels


# df = pd.read_csv("dataset-multiclass.txt")
df = pd.read_csv("dataset-multiclass-fewshot.txt")
df = df.sample(frac=1).reset_index(drop=True)

corpus = df["text"]
y = df["labels"]

predicted_labels = corpus.apply(lambda x: classify_text(x, LABELS))
# predicted_labels = corpus.apply(lambda x: classify_text(x, LABELS_ENGLISH))
# predicted_labels.rename(columns = {'Love poem':'Ljubavna poezija', 'Thought-reflexive poem':'Misaono - refleksivna poezija', 'Patriotic poem':'Rodoljubiva poezija'}, inplace=True)
# predicted_labels = predicted_labels.apply(lambda x: translate_labels(x))
print(f1_score(y, predicted_labels, average='macro'))

cm = confusion_matrix(y, predicted_labels, normalize='true', labels=LABELS)
plt.figure(figsize=(8, 6))
sb.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=True,
            xticklabels=['Ljubavna', 'Misaono-refleksivna', 'Rodoljubiva'], yticklabels=['Ljubavna', 'Misaono-refleksivna', 'Rodoljubiva'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()