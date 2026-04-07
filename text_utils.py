import unicodedata
from difflib import SequenceMatcher


UFS_BRASIL = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
    "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
    "SP", "SE", "TO",
}


def normalizar_texto(texto):
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c)).lower().strip()


def limpar_espacos(texto):
    return " ".join(texto.split())


def dividir_tokens(texto):
    tokens = []
    atual = []
    for ch in texto:
        if ch.isalnum() or ch in {"_"}:
            atual.append(ch)
        else:
            if atual:
                tokens.append("".join(atual))
                atual = []
    if atual:
        tokens.append("".join(atual))
    return tokens


def apenas_digitos(texto):
    return "".join(ch for ch in texto if ch.isdigit())


def apenas_letras(texto):
    return "".join(ch for ch in texto if ch.isalpha())


def similaridade(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def contem_keyword(linha, keywords, limiar=0.82):
    linha_norm = normalizar_texto(linha)
    tokens = dividir_tokens(linha_norm)

    for kw in keywords:
        kw_norm = normalizar_texto(kw)
        if kw_norm in linha_norm:
            return True
        for tok in tokens:
            if similaridade(tok, kw_norm) >= limiar:
                return True
    return False


def tem_contexto_proximo(linhas, indice, keywords, limiar=0.78, janela=1):
    ini = max(0, indice - janela)
    fim = min(len(linhas), indice + janela + 1)
    for j in range(ini, fim):
        if contem_keyword(linhas[j], keywords, limiar=limiar):
            return True
    return False


def extrair_datas_linha(linha):
    datas = []
    s = linha
    n = len(s)

    for i in range(n - 9):
        d1 = s[i : i + 2]
        sep1 = s[i + 2]
        m1 = s[i + 3 : i + 5]
        sep2 = s[i + 5]
        y1 = s[i + 6 : i + 10]

        if not (d1.isdigit() and m1.isdigit() and y1.isdigit()):
            continue
        if sep1 not in {"/", "-", "."} or sep2 not in {"/", "-", "."}:
            continue

        dia = int(d1)
        mes = int(m1)
        ano = int(y1)
        if 1 <= dia <= 31 and 1 <= mes <= 12 and 1900 <= ano <= 2100:
            datas.append(f"{d1}/{m1}/{y1}")

    return datas


def extrair_cpfs_formatados_linha(linha):
    cpfs = []
    s = linha
    n = len(s)

    for i in range(n - 13):
        trecho = s[i : i + 14]
        if not (
            trecho[0:3].isdigit()
            and trecho[3] == "."
            and trecho[4:7].isdigit()
            and trecho[7] == "."
            and trecho[8:11].isdigit()
            and trecho[11] == "-"
            and trecho[12:14].isdigit()
        ):
            continue

        cpf = trecho[0:3] + trecho[4:7] + trecho[8:11] + trecho[12:14]
        cpfs.append(cpf)

    return cpfs


def normalizar_data(data):
    dia = int(data[0:2])
    mes = int(data[3:5])
    ano = int(data[6:10])
    return dia, mes, ano
