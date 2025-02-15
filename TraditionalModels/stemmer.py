import reldi_tokeniser

rules = {
    'ovnicxki': '',
    'ovnicxka': '',
    'ovnika': '',
    'ovniku': '',
    'ovnicxe': '',
    'kujemo': '',
    'ovacyu': '',
    'ivacyu': '',
    'isacyu': '',
    'dosmo': '',
    'ujemo': '',
    'ijemo': '',
    'ovski': '',
    'ajucxi': '',
    'icizma': '',
    'ovima': '',
    'ovnik': '',
    'ognu': '',
    'inju': '',
    'enju': '',
    'cxicyu': '',
    'sxtva': '',
    'ivao': '',
    'ivala': '',
    'ivalo': '',
    'skog': '',
    'ucxit': '',
    'ujesx': '',
    'ucyesx': '',
    'ocyesx': '',
    'osmo': '',
    'ovao': '',
    'ovala': '',
    'ovali': '',
    'ismo': '',
    'ujem': '',
    'esmo': '',
    'asmo': '',
    'zxemo': '',
    'cyemo': '',
    'bemo': '',
    'ovan': '',
    'ivan': '',
    'isan': '',
    'uvsxi': '',
    'ivsxi': '',
    'evsxi': '',
    'avsxi': '',
    'sxucyi': '',
    'uste': '',
    'icxe': 'i',
    'acxe': 'ak',
    'uzxe': 'ug',
    'azxe': 'ag',
    'aci': 'ak',
    'oste': '',
    'aca': '',
    'enu': '',
    'enom': '',
    'enima': '',
    'eta': '',
    'etu': '',
    'etom': '',
    'adi': '',
    'alja': '',
    'nju': 'nj',
    'lju': '',
    'lja': '',
    'lji': '',
    'lje': '',
    'ljom': '',
    'ljama': '',
    'zi': 'g',
    'etima': '',
    'ac': '',
    'becyi': 'beg',
    'nem': '',
    'nesx': '',
    'ne': '',
    'nemo': '',
    'nimo': '',
    'nite': '',
    'nete': '',
    'nu': '',
    'ce': '',
    'ci': '',
    'cu': '',
    'ca': '',
    'cem': '',
    'cima': '',
    'sxcyu': 's',
    'ara': 'r',
    'iste': '',
    'este': '',
    'aste': '',
    'ujte': '',
    'jete': '',
    'jemo': '',
    'jem': '',
    'jesx': '',
    'ijte': '',
    'inje': '',
    'acxki': '',
    'anje': '',
    'inja': '',
    'nog': '',
    'omu': '',
    'emu': '',
    'uju': '',
    'iju': '',
    'sko': '',
    'eju': '',
    'ahu': '',
    'ucyu': '',
    'icyu': '',
    'ecyu': '',
    'acyu': '',
    'ocu': '',
    'izi': 'ig',
    'ici': 'ik',
    'tko': 'd',
    'tka': 'd',
    'ast': '',
    'tit': '',
    'nusx': '',
    'cyesx': '',
    'cxno': '',
    'cxni': '',
    'cxna': '',
    'uto': '',
    'oro': '',
    'eno': '',
    'ano': '',
    'umo': '',
    'smo': '',
    'imo': '',
    'emo': '',
    'ulo': '',
    'sxlo': '',
    'slo': '',
    'ila': '',
    'ilo': '',
    'ski': '',
    'ska': '',
    'elo': '',
    'njo': '',
    'ovi': '',
    'evi': '',
    'uti': '',
    'iti': '',
    'eti': '',
    'ati': '',
    'vsxi': '',
    'ili': '',
    'eli': '',
    'ali': '',
    'uji': '',
    'nji': '',
    'ucyi': '',
    'sxcyi': '',
    'ecyi': '',
    'ucxi': '',
    'oci': '',
    'ove': '',
    'eve': '',
    'ute': '',
    'ste': '',
    'nte': '',
    'kte': '',
    'jte': '',
    'ite': '',
    'ete': '',
    'cyi': '',
    'usxe': '',
    'esxe': '',
    'asxe': '',
    'une': '',
    'ene': '',
    'ule': '',
    'ile': '',
    'ele': '',
    'ale': '',
    'uke': '',
    'tke': '',
    'ske': '',
    'uje': '',
    'tje': '',
    'ucye': '',
    'sxcye': '',
    'icye': '',
    'ecye': '',
    'ucxe': '',
    'oce': '',
    'ova': '',
    'eva': '',
    'ava': 'av',
    'uta': '',
    'ata': '',
    'ena': '',
    'ima': '',
    'ama': '',
    'ela': '',
    'ala': '',
    'aka': '',
    'aja': '',
    'jmo': '',
    'oga': '',
    'ega': '',
    'aća': '',
    'oca': '',
    'aba': '',
    'cxki': '',
    'ju': '',
    'hu': '',
    'cyu': '',
    'ut': '',
    'it': '',
    'et': '',
    'at': '',
    'usx': '',
    'isx': '',
    'esx': '',
    'uo': '',
    'no': '',
    'mo': '',
    'lo': '',
    'io': '',
    'eo': '',
    'ao': '',
    'un': '',
    'an': '',
    'om': '',
    'ni': '',
    'im': '',
    'em': '',
    'uk': '',
    'uj': '',
    'oj': '',
    'li': '',
    'uh': '',
    'oh': '',
    'ih': '',
    'eh': '',
    'ah': '',
    'og': '',
    'eg': '',
    'te': '',
    'sxe': '',
    'le': '',
    'ke': '',
    'ko': '',
    'ka': '',
    'ti': '',
    'he': '',
    'cye': '',
    'cxe': '',
    'ad': '',
    'ecy': '',
    'na': '',
    'ma': '',
    'ul': '',
    'ku': '',
    'la': '',
    'nj': 'nj',
    'lj': 'lj',
    'ha': '',
    'a': '',
    'e': '',
    'u': '',
    'sx': '',
    'o': '',
    'j': '',
    'i': ''
}
dictionary = {
    # biti glagol
    'bih': 'biti',
    'bi': 'biti',
    'bismo': 'biti',
    'biste': 'biti',
    'bisxe': 'biti',
    'budem': 'biti',
    'budesx': 'biti',
    'bude': 'biti',
    'budemo': 'biti',
    'budete': 'biti',
    'budu': 'biti',
    'bio': 'biti',
    'bila': 'biti',
    'bili': 'biti',
    'bile': 'biti',
    'biti': 'biti',
    'bijah': 'biti',
    'bijasxe': 'biti',
    'bijasmo': 'biti',
    'bijaste': 'biti',
    'bijahu': 'biti',
    'besxe': 'biti',
    # jesam
    'sam': 'jesam',
    'si': 'jesam',
    'je': 'jesam',
    'smo': 'jesam',
    'ste': 'jesam',
    'su': 'jesam',
    'jesam': 'jesam',
    'jesi': 'jesam',
    'jeste': 'jesam',
    'jesmo': 'jesam',
    'jesu': 'jesam',
    # glagol hteti
    'cyu': 'hteti',
    'cyesx': 'hteti',
    'cye': 'hteti',
    'cyemo': 'hteti',
    'cyete': 'hteti',
    'hocyu': 'hteti',
    'hocyesx': 'hteti',
    'hocyemo': 'hteti',
    'hocyete': 'hteti',
    'hocye': 'hteti',
    'hteo': 'hteti',
    'htela': 'hteti',
    'hteli': 'hteti',
    'htelo': 'hteti',
    'htele': 'hteti',
    'htedoh': 'hteti',
    'htede': 'hteti',
    'htedosmo': 'hteti',
    'htedoste': 'hteti',
    'htedosxe': 'hteti',
    'hteh': 'hteti',
    'hteti': 'hteti',
    'htejucyi': 'hteti',
    'htevsxi': 'hteti',
    # glagol moći
    'mogu': 'mocyi',
    'možeš': 'mocyi',
    'može': 'mocyi',
    'možemo': 'mocyi',
    'možete': 'mocyi',
    'mogao': 'mocyi',
    'mogli': 'mocyi',
    'moći': 'mocyi'
}


def stem(text):
    text = text.lower()
    text = text.replace("š", "sx")
    text = text.replace("č", "cx")
    text = text.replace("ć", "cy")
    text = text.replace("đ", "dx")
    text = text.replace("ž", "zx")
    text = text.replace("“", "\"")
    text = text.replace("\"", "")
    text = text.replace("”", "\"")
    text = text.replace("'", "\"")
    text = text.replace("’", "\"")
    text = text.replace("€", "eur")
    text = text.replace("„", "\"")
    lam = reldi_tokeniser.run(text, 'sr', bert=True).split()

    i = 0
    for word in lam:
        for key in dictionary:
            if key == word:
                lam[i] = dictionary[key]
                break
        for key in rules:
            # Tokenize only words larger than 2 characters, apart from modal verbs
            if word.endswith(key) and len(word) - len(key) > 2:
                lam[i] = word[:-len(key)] + rules[key]
                break
        i = i + 1
    end_str = ""
    for word in lam:
        end_str = end_str + " " + word
    end_str = end_str.strip()

    end_str = end_str.replace("sx", "š")
    end_str = end_str.replace("cx", "č")
    end_str = end_str.replace("cy", "ć")
    end_str = end_str.replace("dx", "đ")
    end_str = end_str.replace("zx", "ž")

    # end_str = end_str.split()
    return end_str


