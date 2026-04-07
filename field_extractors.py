import re

from text_utils import (
    UFS_BRASIL,
    apenas_digitos,
    apenas_letras,
    contem_keyword,
    dividir_tokens,
    extrair_cpfs_formatados_linha,
    extrair_datas_linha,
    limpar_espacos,
    normalizar_data,
    normalizar_texto,
    similaridade,
    tem_contexto_proximo,
)


def validar_cpf(cpf):
    if len(cpf) != 11 or len(set(cpf)) == 1:
        return False

    soma1 = 0
    for i in range(9):
        soma1 += int(cpf[i]) * (10 - i)
    dig1 = (soma1 * 10) % 11
    if dig1 == 10:
        dig1 = 0
    if dig1 != int(cpf[9]):
        return False

    soma2 = 0
    for i in range(10):
        soma2 += int(cpf[i]) * (11 - i)
    dig2 = (soma2 * 10) % 11
    if dig2 == 10:
        dig2 = 0
    return dig2 == int(cpf[10])


def formatar_cpf(cpf_numerico):
    return f"{cpf_numerico[0:3]}.{cpf_numerico[3:6]}.{cpf_numerico[6:9]}-{cpf_numerico[9:11]}"


def extrair_cpf(texto, linhas):
    candidatos = []

    # Prioridade absoluta: token explicitamente ancorado em CPF.
    for i, linha in enumerate(linhas):
        linha_up = linha.upper()
        tokens = dividir_tokens(linha_up)
        contexto_cpf = tem_contexto_proximo(linhas, i, ["cpf", "cadastro pessoa fisica"], limiar=0.74, janela=1)

        for t_idx, tok in enumerate(tokens):
            if not tok.startswith("CPF"):
                continue

            # Caso OCR venha como CPF13810533602.
            digitos_tok = apenas_digitos(tok)
            if len(digitos_tok) >= 11:
                candidatos.append((digitos_tok[:11], 9.0 if contexto_cpf else 7.0))

            # Caso OCR venha separado: CPF 13810533602.
            if t_idx + 1 < len(tokens):
                prox = apenas_digitos(tokens[t_idx + 1])
                if len(prox) >= 11:
                    candidatos.append((prox[:11], 8.8 if contexto_cpf else 6.8))

    # Prioridade maxima: CPF ja no formato xxx.xxx.xxx-xx.
    for i, linha in enumerate(linhas):
        contexto_cpf = tem_contexto_proximo(linhas, i, ["cpf", "cadastro pessoa fisica"], limiar=0.74, janela=1)
        contexto_rg = tem_contexto_proximo(linhas, i, ["rg", "registro geral", "identidade"], limiar=0.72, janela=1)
        for cpf in extrair_cpfs_formatados_linha(linha):
            score = 4.0
            if validar_cpf(cpf):
                score += 3.0
            if contexto_cpf:
                score += 2.0
            if contexto_rg and not contexto_cpf:
                score -= 1.0
            candidatos.append((cpf, score))

    # Prioriza linhas com contexto de CPF.
    for i, linha in enumerate(linhas):
        linha_norm = normalizar_texto(linha).replace(" ", "")
        peso = 0.0
        contexto_cpf = tem_contexto_proximo(linhas, i, ["cpf", "cadastro pessoa fisica"], limiar=0.74, janela=1)
        contexto_rg = tem_contexto_proximo(linhas, i, ["rg", "registro geral", "identidade"], limiar=0.72, janela=1)

        if "cpf" in linha_norm:
            peso += 1.2
        if contexto_cpf:
            peso += 0.8
        if contexto_rg and not contexto_cpf:
            peso -= 0.8

        bloco = linha
        if i + 1 < len(linhas):
            bloco = f"{bloco} {linhas[i + 1]}"

        digitos = apenas_digitos(bloco)
        for j in range(0, max(0, len(digitos) - 10)):
            trecho = digitos[j : j + 11]
            if len(trecho) == 11:
                candidatos.append((trecho, peso))

    # Fallback global quando label falha por OCR.
    digitos_globais = apenas_digitos(texto)
    for j in range(0, max(0, len(digitos_globais) - 10)):
        trecho = digitos_globais[j : j + 11]
        if len(trecho) == 11:
            candidatos.append((trecho, 0.2))

    melhor = None
    melhor_score = -1
    for cpf, peso in candidatos:
        score = peso
        if validar_cpf(cpf):
            score += 3.0
        if score > melhor_score:
            melhor = cpf
            melhor_score = score

    if melhor:
        return formatar_cpf(melhor)

    return None


def parece_matricula(texto):
    t_norm = normalizar_texto(texto)
    padroes_matricula = ["matricula", "registro", "codigo", "uso", "doc uso"]
    for padrao in padroes_matricula:
        if padrao in t_norm:
            return True

    digitos = apenas_digitos(texto)
    letras = apenas_letras(texto)
    if len(digitos) >= 10 and len(letras) == 0:
        return True
    if len(digitos) >= 8 and len(letras) <= 2:
        return True

    return False


def normalizar_rg(valor):
    return "".join(ch for ch in valor.upper() if ch.isalnum())


def extrair_rgs_formatados_linha(linha, cpf_norm_para_comparacao=None):
    rgs = []

    # Padrao 1: UF + digitos + opcional sufixo (ex.: MG21469017SSP).
    # Usa tokenização para evitar pegar substring dentro de CPF.
    for tok in dividir_tokens(linha.upper()):
        if len(tok) < 8:
            continue
        uf = tok[:2]
        if uf not in UFS_BRASIL:
            continue

        resto = tok[2:]
        prefixo_dig = []
        sufixo = []
        mudou = False
        for ch in resto:
            if ch.isdigit() and not mudou:
                prefixo_dig.append(ch)
            elif ch.isalpha():
                mudou = True
                sufixo.append(ch)
            else:
                prefixo_dig = []
                sufixo = []
                break

        if not prefixo_dig:
            continue
        if not (6 <= len(prefixo_dig) <= 10):
            continue
        if len(sufixo) > 4:
            continue

        rgs.append(("estado", tok))

    # Padrao 2: XX.XXX.XXX-X (numeros com pontos e hifen)
    # Mais genérico, pode confundir com CPF
    padrao_pontuado = r"(?<!\d)\d{2}\.\d{3}\.\d{3}-\d{1,2}(?!\d)"
    matches = re.finditer(padrao_pontuado, linha, re.IGNORECASE)
    for match in matches:
        rg_str = match.group()
        rg_norm = normalizar_rg(rg_str)

        # Filtrar se for substring do CPF normalizado
        if cpf_norm_para_comparacao and rg_norm in cpf_norm_para_comparacao:
            continue

        rgs.append(("pontuado", rg_str))

    return rgs


def extrair_rg(texto, linhas, cpf_extraido=None):
    candidatos = []

    # Normalizar CPF extraído para comparação
    cpf_norm = None
    if cpf_extraido:
        cpf_norm = normalizar_rg(cpf_extraido)  # Remove . e - do CPF para comparação

    # Prioridade absoluta: token explicitamente ancorado em RG.
    for i, linha in enumerate(linhas):
        tokens = dividir_tokens(linha.upper())
        contexto_rg = tem_contexto_proximo(linhas, i, ["rg", "registro geral", "identidade", "registro"], limiar=0.72, janela=1)

        for t_idx, tok in enumerate(tokens):
            if tok.startswith("RG"):
                dig = apenas_digitos(tok)
                if 6 <= len(dig) <= 10:
                    candidatos.append((dig, 16.5 if contexto_rg else 14.5))

            if tok == "RG" and t_idx + 1 < len(tokens):
                prox = normalizar_rg(tokens[t_idx + 1])
                dig = apenas_digitos(prox)
                if 6 <= len(dig) <= 10:
                    candidatos.append((prox, 16.2 if contexto_rg else 14.2))

    # Prioridade MAXIMA: RG com padrão de estado (ex: MG21469017SSP)
    # Estes têm score muito alto e dificilmente se confundem com CPF
    for i, linha in enumerate(linhas):
        contexto_rg = tem_contexto_proximo(linhas, i, ["rg", "registro geral", "identidade", "registro"], limiar=0.72, janela=1)
        contexto_cpf = tem_contexto_proximo(linhas, i, ["cpf", "cadastro pessoa fisica"], limiar=0.74, janela=1)
        rgs_encontrados = extrair_rgs_formatados_linha(linha, cpf_norm)
        for tipo_padrao, rg_formato in rgs_encontrados:
            if tipo_padrao == "estado":  # Padrão com estado
                score = 15.0  # Score extremamente alto para estado
                if contexto_rg:
                    score += 3.0
                if contexto_cpf and not contexto_rg:
                    score -= 1.5
                candidatos.append((rg_formato, score))

    # Segunda prioridade: RG com pontuação (XX.XXX.XXX-X) com contexto forte
    for i, linha in enumerate(linhas):
        contexto_rg = tem_contexto_proximo(linhas, i, ["rg", "registro geral", "identidade", "registro"], limiar=0.72, janela=1)
        contexto_cpf = tem_contexto_proximo(linhas, i, ["cpf", "cadastro pessoa fisica"], limiar=0.74, janela=1)
        rgs_encontrados = extrair_rgs_formatados_linha(linha, cpf_norm)
        for tipo_padrao, rg_formato in rgs_encontrados:
            if tipo_padrao == "pontuado":  # Padrão pontuado (já filtrado de CPF)
                score = 8.0  # Score bem alto, mas menor que estado
                if contexto_rg:
                    score += 2.5
                if contexto_cpf and not contexto_rg:
                    score -= 3.0
                candidatos.append((rg_formato, score))

    # Terceira prioridade: linhas com contexto de RG (sem formatação).
    for linha in linhas:
        linha_norm = normalizar_texto(linha)
        tokens = dividir_tokens(linha.upper())
        peso = 0.0

        if contem_keyword(linha_norm, ["rg", "registro geral", "identidade", "doc"], limiar=0.72):
            peso += 1.4
        if peso <= 0:
            continue

        for tok in tokens:
            if parece_matricula(tok):
                continue

            rg = normalizar_rg(tok)
            if rg.startswith("CPF"):
                continue

            # Filtrar RG sem formatação que seja idêntico ao CPF normalizado
            if cpf_norm and rg == cpf_norm:
                continue

            # Filtrar RG que seja substring do CPF
            if cpf_norm and rg in cpf_norm:
                continue

            qtd_digitos = sum(ch.isdigit() for ch in rg)
            qtd_letras = sum(ch.isalpha() for ch in rg)
            if len(rg) < 6 or qtd_digitos < 5:
                continue

            # Aceita RG numerico puro (geralmente 7-10 digitos).
            candidato_valido = False
            if qtd_letras == 0 and 6 <= qtd_digitos <= 10:
                candidato_valido = True

            # Aceita RG no formato UF + digitos + sufixo opcional.
            if qtd_letras > 0 and len(rg) >= 8:
                uf = rg[:2]
                if uf in UFS_BRASIL:
                    meio = rg[2:]
                    prefixo_dig = []
                    sufixo = []
                    mudou = False
                    for ch in meio:
                        if ch.isdigit() and not mudou:
                            prefixo_dig.append(ch)
                        elif ch.isalpha():
                            mudou = True
                            sufixo.append(ch)
                        else:
                            prefixo_dig = []
                            sufixo = []
                            break
                    if 6 <= len(prefixo_dig) <= 10 and len(sufixo) <= 4:
                        candidato_valido = True

            if not candidato_valido:
                continue

            score = peso
            if rg[:2].isalpha() and qtd_digitos >= 6:
                score += 0.5
            if len(rg) == 11 and validar_cpf(rg):
                score -= 3.0
            candidatos.append((rg, score))

    if not candidatos:
        return None

    candidatos.sort(key=lambda x: x[1], reverse=True)
    return candidatos[0][0]


def parece_nome_pessoa(texto):
    t = texto.strip()
    if not t:
        return False

    t_norm = normalizar_texto(t)
    bloqueios = {
        "instituicao",
        "ensino",
        "faculdade",
        "curso",
        "serie",
        "documento",
        "carteira",
        "identificacao",
        "estudante",
        "nacional",
        "validade",
        "nascimento",
        "emissao",
        "codigo",
        "cpf",
        "rg",
        "data",
        "educacional",
        "graduacao",
        "dne",
        "filiacao",
        "nacionalidade",
        "habilitacao",
        "driver",
        "license",
        "permiso",
        "conduccion",
    }

    if any(p in t_norm for p in bloqueios):
        return False

    tokens = [tok for tok in t.split() if tok]
    if len(tokens) > 6:
        return False

    # Aceita nome colado em uma palavra (ex.: MARIACLARA) quando bem caracteristico.
    if len(tokens) == 1:
        token = apenas_letras(tokens[0])
        if len(token) < 8 or len(token) > 22:
            return False
        token_norm = normalizar_texto(token)
        if any(chave in token_norm for chave in {"republic", "feder", "brasil", "minister", "carteira"}):
            return False
        return token.isupper()

    if len(tokens) < 2:
        return False

    palavras_validas = 0
    for tok in tokens:
        limpo = apenas_letras(tok)
        if len(limpo) >= 2:
            palavras_validas += 1

    if palavras_validas < 2:
        return False

    comprimentos = [len(apenas_letras(tok)) for tok in tokens]
    return any(c >= 4 for c in comprimentos)


def limpar_nome_candidato(linha):
    tokens = [tok for tok in linha.split() if tok]
    saida = []
    bloqueios_token = {
        "nome",
        "sobrenome",
        "filiacao",
        "nacionalidade",
        "habilitacao",
        "doc",
        "identidad",
        "identidade",
        "emissor",
        "cat",
        "hae",
        "cp",
        "cnh",
        "brasileiro",
        "brasileira",
        "carteira",
        "nacional",
        "registro",
        "validade",
        "universidade",
        "curso",
        "estado",
        "minas",
        "gerais",
        "uemg",
    }
    bloqueios_substring = {
        "univers",
        "faculd",
        "institu",
        "curso",
        "uemg",
        "federal",
        "estad",
    }
    for tok in tokens:
        tok_norm = normalizar_texto(tok)
        if any(ch.isdigit() for ch in tok):
            continue

        if any(chave in tok_norm for chave in bloqueios_substring):
            continue

        bloqueado = tok_norm in bloqueios_token
        if not bloqueado:
            for item in bloqueios_token:
                if similaridade(tok_norm, item) >= 0.86:
                    bloqueado = True
                    break

        if bloqueado:
            continue

        limpo = apenas_letras(tok)
        if len(limpo) >= 3:
            saida.append(limpo)

    if not saida:
        return ""
    return " ".join(saida)


def cortar_ao_primeiro_digito(texto):
    for i, ch in enumerate(texto):
        if ch.isdigit():
            return texto[:i].strip()
    return texto.strip()


def truncar_ao_campo(texto):
    tokens = [tok for tok in texto.split() if tok]
    if not tokens:
        return ""

    campos_corte = [
        "carteira",
        "republica",
        "ministerio",
        "filiacao",
        "nacionalidade",
        "cpf",
        "rg",
        "validade",
        "registro",
        "doc",
        "identidade",
        "emissor",
    ]

    saida = []
    for tok in tokens:
        tok_norm = normalizar_texto(tok)
        deve_cortar = False
        for campo in campos_corte:
            if campo in tok_norm or similaridade(tok_norm, campo) >= 0.82:
                deve_cortar = True
                break

        if deve_cortar:
            break
        saida.append(tok)

    return " ".join(saida)


def eh_label_nome(linha):
    linha_norm = limpar_espacos(normalizar_texto(linha))
    if linha_norm in {"nome", "nome completo", "nome do aluno", "aluno"}:
        return True
    return contem_keyword(linha_norm, ["nome"], limiar=0.9)


def extrair_nome_por_contexto(linhas):
    candidatos = []

    for i, linha in enumerate(linhas):
        if not (eh_label_nome(linha) or contem_keyword(linha, ["nome", "sobrenome"], limiar=0.75)):
            continue

        tokens_linha = [tok for tok in linha.split() if tok]
        for idx, tok in enumerate(tokens_linha):
            tok_norm = normalizar_texto(tok)
            if "nome" not in tok_norm and similaridade(tok_norm, "nome") < 0.85:
                continue

            # Caso "... THIAGO LUIZ PEREIRA Nome:" (label apos o valor)
            antes = " ".join(tokens_linha[max(0, idx - 4):idx])
            antes_limpo = limpar_nome_candidato(truncar_ao_campo(cortar_ao_primeiro_digito(antes)))
            if parece_nome_pessoa(antes_limpo):
                candidatos.append((antes_limpo, 3.1))

            # Caso "Nome: THIAGO LUIZ PEREIRA" (label antes do valor)
            depois = " ".join(tokens_linha[idx + 1 : idx + 6])
            depois_limpo = limpar_nome_candidato(truncar_ao_campo(cortar_ao_primeiro_digito(depois)))
            if parece_nome_pessoa(depois_limpo):
                candidatos.append((depois_limpo, 3.0))

        # Em alguns documentos o nome aparece na mesma linha do label, com ruido depois.
        atual = truncar_ao_campo(cortar_ao_primeiro_digito(linha))
        atual_limpo = limpar_nome_candidato(atual)
        if parece_nome_pessoa(atual_limpo):
            candidatos.append((atual_limpo, 2.8))

        vizinhos = []
        if i > 0:
            vizinhos.append((linhas[i - 1], 1.2))
        if i + 1 < len(linhas):
            vizinhos.append((linhas[i + 1], 2.3))
        if i + 2 < len(linhas):
            vizinhos.append((linhas[i + 2], 1.0))

        for texto_candidato, peso in vizinhos:
            sem_data = truncar_ao_campo(cortar_ao_primeiro_digito(texto_candidato))
            nome_limpo = limpar_nome_candidato(sem_data)
            if parece_nome_pessoa(nome_limpo):
                candidatos.append((nome_limpo, peso))

    if not candidatos:
        return None

    candidatos.sort(key=lambda x: x[1], reverse=True)
    return candidatos[0][0]


def extrair_datas_por_contexto(linhas):
    candidatos = []
    for i, linha in enumerate(linhas):
        datas_linha = extrair_datas_linha(linha)
        if not datas_linha:
            continue

        contexto = normalizar_texto(linha)
        if i > 0:
            contexto = f"{contexto} {normalizar_texto(linhas[i - 1])}"
        if i + 1 < len(linhas):
            contexto = f"{contexto} {normalizar_texto(linhas[i + 1])}"

        for pos, data in enumerate(datas_linha):
            score_nasc = 0.0
            score_valid = 0.0
            score_emi = 0.0

            if contem_keyword(contexto, ["nasc", "nascimento", "birth"], limiar=0.72):
                score_nasc += 3.0
            if contem_keyword(contexto, ["validade", "valido", "venc", "expire"], limiar=0.72):
                score_valid += 3.0
            if contem_keyword(contexto, ["emissao", "emitido", "expedicao", "issue"], limiar=0.72):
                score_emi += 3.0

            if contem_keyword(contexto, ["nome", "sobrenome", "habilitacao"], limiar=0.72):
                score_valid -= 1.5
                score_emi -= 1.0

            _, _, ano = normalizar_data(data)
            if ano <= 2010:
                score_nasc += 1.4
            if ano >= 2025:
                score_valid += 0.8

            if len(datas_linha) >= 2:
                if pos == 0:
                    score_emi += 0.9
                if pos == 1:
                    score_valid += 0.9

            candidatos.append((data, "nascimento", score_nasc))
            candidatos.append((data, "validade", score_valid))
            candidatos.append((data, "emissao", score_emi))

    datas = {}
    usados = set()
    for campo in ["nascimento", "emissao", "validade"]:
        melhores = [c for c in candidatos if c[1] == campo and c[0] not in usados]
        if not melhores:
            continue
        melhores.sort(key=lambda x: x[2], reverse=True)
        data, _, score = melhores[0]
        if score > 0.2:
            datas[campo] = data
            usados.add(data)

    if "nascimento" not in datas and candidatos:
        unicos = sorted({c[0] for c in candidatos}, key=lambda d: normalizar_data(d)[2])
        if unicos:
            datas["nascimento"] = unicos[0]

    if "validade" not in datas and candidatos:
        unicos = sorted({c[0] for c in candidatos}, key=lambda d: normalizar_data(d)[2], reverse=True)
        if unicos:
            for data in unicos:
                if data not in datas.values():
                    datas["validade"] = data
                    break

    if "emissao" not in datas and candidatos:
        unicos = sorted({c[0] for c in candidatos}, key=lambda d: normalizar_data(d)[2])
        for data in unicos:
            if data not in datas.values():
                datas["emissao"] = data
                break

    return datas
