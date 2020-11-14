import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import wikipedia as wiki
import nltk
import re
from collections import OrderedDict

class RespondedorPerguntas:
    def __init__(self, pretrained_model_path='squad-pt-saida'):
        wiki.set_lang("pt")
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model_path)
        self.max_len = self.model.config.max_position_embeddings
     
    '''
    Realiza tokenizacao retornando uma lista de chunks
    '''
    def __tokenize(self, pergunta, contexto):
        inputs = self.tokenizer.encode_plus(pergunta, contexto, add_special_tokens=True, return_tensors="pt")

        # Caso o input seja grande demais para o modelo (mais de 512 caracteres)
        if len(inputs["input_ids"].tolist()[0]) > self.max_len:
            inputs = self.__chunkify(inputs)
        else:
            inputs = [inputs]

        return inputs

    '''
    Realiza chunkfy
    '''
    def __chunkify(self, inputs):
        # Cria máscara de pergunta (0 é pergunta e 1 é contexto)
        mascara_pergunta = inputs['token_type_ids'].lt(1)
        pergunta = torch.masked_select(inputs['input_ids'], mascara_pergunta)

        # Tamanho do chunk considerando já o token final [SEP]
        tamanho_chunk = self.max_len - pergunta.size()[0] - 1 

        # Cria dict de dict que será povoado com cada chunk de contexto:
        chunked_input = OrderedDict()
        for k,v in inputs.items():
            # Separa pergunta e contexto:
            pergunta = torch.masked_select(v, mascara_pergunta)
            contexto = torch.masked_select(v, ~mascara_pergunta)
            # Faz o split do contexto e itera sobre eles:
            chunks = torch.split(contexto, tamanho_chunk)
            for i, chunk in enumerate(chunks):
                # Concatena pergunta e contexto:
                elem = torch.cat((pergunta, chunk))
                # Apenas se não for o último chunk
                if i != len(chunks)-1:
                    if k == 'input_ids':
                        # Concatena id de token final SEP
                        elem = torch.cat((elem, torch.tensor([102])))
                    else:
                        # Concatena 1 para contexto
                        elem = torch.cat((elem, torch.tensor([1])))

                if i not in chunked_input:
                    chunked_input[i] = {}
                # Gera tensor linha:
                chunked_input[i][k] = torch.unsqueeze(elem, dim=0)
        return list(chunked_input.values())

    '''
    Obtém a resposta de uma pergunta dada um contexto com o score ideal:
    '''
    def __obtem_resposta(self, chunk):
        # Obtém os scores:
        resposta_inicio_scores, resposta_fim_scores = self.model(**chunk)
        resposta_inicio = torch.argmax(resposta_inicio_scores)
        resposta_fim = torch.argmax(resposta_fim_scores) + 1
        resposta = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(chunk["input_ids"][0][resposta_inicio:resposta_fim]))

        # Obtém a melhor resposta:
        return float(torch.max(resposta_inicio_scores)), float(torch.max(resposta_fim_scores)), resposta

    '''
    Busca páginas da Wikipedia
    TODO: Melhorar busca na wikipedia
    '''
    def __buscaPaginas(self, pergunta, results=5):
        return [wiki.page(p) for p in wiki.search(pergunta, results)]

    '''
    Obtém resposta das N melhores respostas juntamente com as P páginas relacionadas:
    '''
    def obtem_respostas(self, pergunta, numero_paginas=5, top=5, debug=False):
        paginas = self.__buscaPaginas(pergunta, numero_paginas)
        respostas = []
        for ordem_pagina, pagina in enumerate(paginas):
            chunks = self.__tokenize(pergunta, pagina.content)
            if debug:
                print(f"Buscando resposta na página: {pagina.title} - {len(chunks)} trechos de texto")
            for chunk in chunks:
                r = self.__obtem_resposta(chunk)
                if r[2] and r[2] != '[CLS]':
                    # Calcula score pela fórmula (score_inicio + score_fim) * (numero_paginas - ordem_pagina):
                    respostas.append(tuple([pagina.title, (r[0]+r[1])*(numero_paginas-ordem_pagina), r[2]]))
            if debug:
                print(f"Número de respostas identificadas: {len(respostas)}")

        # Obtém as Top N respostas ordenadas:
        return sorted(respostas, key=lambda tup: tup[1], reverse=True)[0:top]