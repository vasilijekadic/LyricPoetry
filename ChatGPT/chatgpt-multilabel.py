from openai import OpenAI
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, multilabel_confusion_matrix
import seaborn as sb
from matplotlib import pyplot as plt
import numpy as np
import ast

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
    # prompt = (f"This is a multilabel classification task."
    #           f"Classify the following poem written in Serbian into the appropriate categories from the list: {labels}."
    #           f"A poem can have multiple labels. Your response should be only the list of labels, for example: ['Love poem', 'Patriotic poem'], and without any additional text."
    #           f"Here are some examples:\n"
    #           f"Poem: Prelepa reko! dok tvoj pramen kristalno čist treperi, ti žarke si lepote znamen – srce neskrito ljupko – smelih draži klupko u starog Alberta kćeri; Kad ona tvom se valu smeši – što trepti kroz odsjaje – tad od svih potoka lepši njen poklonik postaje; jer, kao tvoj val, on sred srca lik čuva njen duboko – i drhti dok se nad njim zrca duše joj željno oko.\n Labels: ['Love poem']\n\n"
    #           f"Poem: Kao kad zaroniš do dna mora, A samo kamenčić zgrneš u dlan, Tako ti počne neka zora, Tako se završi po neki dan. I suze detinje kaplju sve teže Na snove prazne k’o prazne mreže. A nekad onako, kao od šale, Padne pred tebe zvezda prava, A ti je šutneš vrhom sandale I odeš dalje. I kad se spava, Kad nebo zaljulja sva svoj klatna, Čitava noć ti od snova – zlatna. Posle detinjstva šta se sve menja? Opet se ređaju snovi i snovi. Između zvezda i kamenja Jastuk kroz život i dalje plovi. Samo je nešto teža glava. Manje se sanja – više se spava.  \n Labels: ['Thought-reflexive poem']\n\n"
    #           f"Poem: Na mekom, toplom krilu Jedinka sina njiha, Ljubi ga noć i dan; Cvećem mu vlasi kiti, Pesmom mu sladi san. Raste joj sinak, raste, Na svoje noge staje, Vasceo majčin svet – Oh, nije šala, nije: Na grani jedan cvet. Raste joj sinak, raste, A majka dršće, strepi, U nežnom srcu svom, Da ne bi na cvet pao Iz vedra neba grom. Raste joj sinak, raste, Jedino blago majci Koje joj dade Bog. Ko ne bi brižno čuvo „Zenicu oka svog?“ Raste joj sinak, raste, Majka bi zvezde s neba Skidala svaki čas, Da sinku, svom jedinku, Od zvezda splete pas. Sinak se snagom paše, A majka sneva svate, Veselja nada svog; Topi se od milina Kraj sina jedinog. Al’ začu s’ bojna truba: „Za Srpstvo, za slobodu!“ – Majka mu paše mač. A kad je pao, niko Nije joj čuo plač.  \n Labels: ['Patriotic poem']\n\n"
    #           f"Poem: Ko sam? Šta sam? Ja sam samo sanjar, Čiji pogled gasne u magli i memli, Živio sam usput, ko da sanjam, Kao mnogi drugi ljudi na toj zemlji. I tebe sad ljubim po navici ,dete, Zato što sam mnoge ljubio, bolećiv, Zato usput, ko što palim cigarete, Govorim i šapćem zaljubljene reči. „Draga moja“,“mila“,“znaj, doveka“ A u duši vazda ista pustoš zrači; Ako dirneš strast u čovekovu biću Istine, bez sumnje, nikad nećeš naći. Zato moja duša ne zna što je jeza Odbijenih želja, neshvaćene tuge. Ti si, moja gipka, lakonoga breza, Stvorena za mene i za mnoge druge. Ali ako tražeć neku srodnu dušu, Vezan protiv želje, utonem u seti, Nikad necu da te ljubomorom gušim, Nikad necu tebe grditi ni kleti. Ko sam? Šta sam? Ja sam samo sanjar, Čiji pogled gasne u magli i memli, I volim te usput, ko da sanjam, Kao mnoge druge na toj zemlji.\n Labels: ['Love poem', 'Thought-reflexive poem']\n\n"
    #           f"Poem: Tiha noćca nastupila, Poče blago širiti se, Nebo plavo, bez oblačka, Stade zv’jezdam’ kititi se. Gledao sam kako milo Svaka svoje zrake daje, Al’ sam gled’o najradije Jato od njih, što tu sjaje. Uporedo sve su stale, Vođa im je prva bila, Pa su nebom putovale Šireć’ svoja zlatna krila. Tad pomislih: mili Bože, Srbadija kad će tako, Poći skupa? A odgovor: – Kad neslogu sprži pak’o… \n Labels: ['Thought-reflexive poem', 'Patriotic poem']\n\n"
    #           f"Poem: {text}\nLabels: ")

    prompt = (f"Ovo je zadatak klasifikacije sa više labela."
              f"Za datu pesmu napisanu na srpskom jeziku odredi njenu vrstu, odnosno odgovarajuću labelu iz liste labela: {labels}."
              f"Pesma može da ima više labela u isto vreme. Tvoj odgovor treba da bude samo lista dodeljenih labela, na primer: ['Ljubavna poezija', 'Rodoljubiva poezija'], bez ikakvog dodatnog teksta."
              f"Evo nekoliko primera:\n"
              f"Pesma: Prelepa reko! dok tvoj pramen kristalno čist treperi, ti žarke si lepote znamen – srce neskrito ljupko – smelih draži klupko u starog Alberta kćeri; Kad ona tvom se valu smeši – što trepti kroz odsjaje – tad od svih potoka lepši njen poklonik postaje; jer, kao tvoj val, on sred srca lik čuva njen duboko – i drhti dok se nad njim zrca duše joj željno oko.\n Labele: ['Ljubavna poezija']\n\n"
              f"Pesma: Kao kad zaroniš do dna mora, A samo kamenčić zgrneš u dlan, Tako ti počne neka zora, Tako se završi po neki dan. I suze detinje kaplju sve teže Na snove prazne k’o prazne mreže. A nekad onako, kao od šale, Padne pred tebe zvezda prava, A ti je šutneš vrhom sandale I odeš dalje. I kad se spava, Kad nebo zaljulja sva svoj klatna, Čitava noć ti od snova – zlatna. Posle detinjstva šta se sve menja? Opet se ređaju snovi i snovi. Između zvezda i kamenja Jastuk kroz život i dalje plovi. Samo je nešto teža glava. Manje se sanja – više se spava.  \n Labele: ['Misaono - refleksivna poezija']\n\n"
              f"Pesma: Na mekom, toplom krilu Jedinka sina njiha, Ljubi ga noć i dan; Cvećem mu vlasi kiti, Pesmom mu sladi san. Raste joj sinak, raste, Na svoje noge staje, Vasceo majčin svet – Oh, nije šala, nije: Na grani jedan cvet. Raste joj sinak, raste, A majka dršće, strepi, U nežnom srcu svom, Da ne bi na cvet pao Iz vedra neba grom. Raste joj sinak, raste, Jedino blago majci Koje joj dade Bog. Ko ne bi brižno čuvo „Zenicu oka svog?“ Raste joj sinak, raste, Majka bi zvezde s neba Skidala svaki čas, Da sinku, svom jedinku, Od zvezda splete pas. Sinak se snagom paše, A majka sneva svate, Veselja nada svog; Topi se od milina Kraj sina jedinog. Al’ začu s’ bojna truba: „Za Srpstvo, za slobodu!“ – Majka mu paše mač. A kad je pao, niko Nije joj čuo plač.  \n Labele: ['Rodoljubiva poezija']\n\n"
              f"Pesma: Ko sam? Šta sam? Ja sam samo sanjar, Čiji pogled gasne u magli i memli, Živio sam usput, ko da sanjam, Kao mnogi drugi ljudi na toj zemlji. I tebe sad ljubim po navici ,dete, Zato što sam mnoge ljubio, bolećiv, Zato usput, ko što palim cigarete, Govorim i šapćem zaljubljene reči. „Draga moja“,“mila“,“znaj, doveka“ A u duši vazda ista pustoš zrači; Ako dirneš strast u čovekovu biću Istine, bez sumnje, nikad nećeš naći. Zato moja duša ne zna što je jeza Odbijenih želja, neshvaćene tuge. Ti si, moja gipka, lakonoga breza, Stvorena za mene i za mnoge druge. Ali ako tražeć neku srodnu dušu, Vezan protiv želje, utonem u seti, Nikad necu da te ljubomorom gušim, Nikad necu tebe grditi ni kleti. Ko sam? Šta sam? Ja sam samo sanjar, Čiji pogled gasne u magli i memli, I volim te usput, ko da sanjam, Kao mnoge druge na toj zemlji.\n Labele: ['Ljubavna poezija', 'Misaono - refleksivna poezija']\n\n"
              f"Pesma: Tiha noćca nastupila, Poče blago širiti se, Nebo plavo, bez oblačka, Stade zv’jezdam’ kititi se. Gledao sam kako milo Svaka svoje zrake daje, Al’ sam gled’o najradije Jato od njih, što tu sjaje. Uporedo sve su stale, Vođa im je prva bila, Pa su nebom putovale Šireć’ svoja zlatna krila. Tad pomislih: mili Bože, Srbadija kad će tako, Poći skupa? A odgovor: – Kad neslogu sprži pak’o… \n Labele: ['Misaono - refleksivna poezija', 'Rodoljubiva poezija']\n\n"
              f"Pesma: {text}\nLabele: ")

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


# df = pd.read_csv("LyricPoetry-multilabel.txt")
df = pd.read_csv("LyricPoetry-multilabel-fewshot.txt")
mlb = MultiLabelBinarizer()
mlb_result = mlb.fit_transform([str(df.loc[i, 'labels']).split(',') for i in range(len(df))])
df = pd.concat([df['text'], pd.DataFrame(mlb_result, columns=list(mlb.classes_))], axis=1)
df = df.sample(frac=1).reset_index(drop=True)

corpus = df["text"]
y = df.loc[:, "Ljubavna poezija":"Rodoljubiva poezija"]

predicted_labels = corpus.apply(lambda x: classify_text(x, LABELS))
# predicted_labels = corpus.apply(lambda x: classify_text(x, LABELS_ENGLISH))
predicted_labels = predicted_labels.apply(lambda x: ast.literal_eval(x))

mlb = MultiLabelBinarizer()
mlb_result = mlb.fit_transform(predicted_labels)
predicted_labels = pd.DataFrame(mlb_result, columns=mlb.classes_)
# predicted_labels.rename(columns = {'Love poem':'Ljubavna poezija', 'Thought-reflexive poem':'Misaono - refleksivna poezija', 'Patriotic poem':'Rodoljubiva poezija'}, inplace=True)
# predicted_labels = predicted_labels.apply(lambda x: translate_labels(x))
predicted_labels = predicted_labels[['Ljubavna poezija', 'Misaono - refleksivna poezija', 'Rodoljubiva poezija']]
print(f1_score(y, predicted_labels, average='macro'))

conf_matrices = multilabel_confusion_matrix(y, predicted_labels)
normalized_confusion_matrices = np.zeros_like(conf_matrices, dtype=float)

for i in range(conf_matrices.shape[0]):
    row_sums = conf_matrices[i].sum(axis=1, keepdims=True)
    normalized_confusion_matrices[i] = conf_matrices[i] / row_sums

fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 5))

for i in range(3):
    sb.heatmap(normalized_confusion_matrices[i], annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=axes[i])
    axes[i].set_title(f'{LABELS[i]}')
    axes[i].set_xlabel('Predicted Label')
    axes[i].set_ylabel('True Label')

plt.tight_layout()
plt.show()
