import os, shutil, tqdm, re
from bs4 import BeautifulSoup
import requests
from ckip_transformers import __version__
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np

def remove_metadata(content):
    metadata_start = "作者"
    metadata_end = "時間"
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    pattern = r"\b(?:\d{1,2}\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:\s+\d{4})?\s+(?:\d{1,2}:){2}\d{1,2}\b"
    content = re.sub(pattern, "", content)

    # 刪除網址
    content = re.sub(r"http\S+|www\S+|https\S+", "", content, flags=re.MULTILINE)
    # 刪除特殊字元
    content = re.sub(r"[^\w\s]+", "", content)

    content = re.sub(f"{metadata_start}.*{metadata_end}", "", content)
    #刪除星期幾
    for weekday in weekdays:
        content = content.replace(weekday, "")
    return content


def get_request(url):
    response = requests.get(url, cookies={'over18': '1'})
    content = response.content
    soup = BeautifulSoup(content, 'html.parser')
    return soup

def get_article_content(url):
    soup = get_request(url)
    article_content = soup.select_one('#main-content').text.split('--')[0]
    article_content = remove_metadata(article_content).strip('\n')

    return article_content

def get_replies(url):
    soup = get_request(url)
    replies = soup.select('.push')
    push_contents = [] #蒐集所有contents後續作分析
    if len(replies) == 0:
        return []
    else:
        for reply in replies:
            #push_tag = reply.select_one('.push-tag').text.strip()
            #push_userid = reply.select_one('.push-userid').text.strip()
            push_content = reply.select_one('.push-content').text.strip()
            push_content = push_content.strip(':').strip()
            push_content = re.sub(r"\s+", "", push_content)
            push_contents.append(push_content)
            

            #print("推文標籤:", push_tag)
            #print("使用者ID:", push_userid)
            #print("回覆內容:", push_content)    
    return push_contents


def calculate_worth_reply(push_contents, article_content):

    ws_article = ws_driver([article_content])  #文章斷詞
    ws_push_contents = ws_driver(push_contents)  #回覆斷詞

    article_words = ws_article[0]  # 文章斷詞表
    push_words = [word for ws in ws_push_contents for word in ws]  # 回覆斷詞表

    article_word_counts = Counter(article_words)  # 統計文章裡面斷詞出現次數

    best_reply = None
    best_count = 0

    for push_content in push_contents:
        push_word_count = sum([article_word_counts.get(word, 0) for word in ws_driver([push_content])[0]])
        # 統計文章單辭出現次數和回覆單辭出現次數
        if push_word_count > best_count:
            best_count = push_word_count
            best_reply = push_content

    if best_reply == None:
        rand_int = np.random.randint(0,len(push_contents))
        best_reply = push_contents[rand_int]

    return best_reply


    
if __name__ == '__main__':
    
    #ckip斷詞
    print("Initializing drivers ... WS")
    ws_driver = CkipWordSegmenter(model="albert-base", device=0)
    print("Initializing drivers ... all done")

    CURRENT_PATH = os.path.dirname(__file__)
    data_path = f'{CURRENT_PATH}/data'
    

    index_value = 35500
    ignore_type = ['[協尋]','[公告]','(本文已被刪除)']
    vectorizer = TfidfVectorizer()
    punctuation_pattern = r'[^\w\s]'

    while index_value>=29000:
        print('index_valuse:',index_value)
        url = f"https://www.ptt.cc/bbs/Gossiping/index{index_value}.html"
        
        soup = get_request(url)
        articles = soup.select('.r-ent')
        
        #取所有的title
        for article in articles:

            title = article.select_one('.title').text.strip()
            title_split = re.split('[ | ]',title)
            title_complete = ''
            
            #如果是要避開的標題confirm_flag=False
            confirm_flag = True
            for word in title_split:
                if word in ignore_type:
                    confirm_flag = False

            for w in title_split:
                if len(w.split('['))> 1 or w =='Re:' :
                    continue
                else:
                    title_complete = title_complete + w

            clean_title = re.sub(punctuation_pattern, '', title_complete, flags=re.IGNORECASE) 
            clean_title = re.sub(r"\s+", "", clean_title)
            
            print('標題:',clean_title)

            if confirm_flag:
                try:
                    #該標題文章連結
                    link = "https://www.ptt.cc" + article.select_one('.title a')['href']
                    push_contents = get_replies(link)
                    article_content = get_article_content(link)
                except:
                    continue
                #前往連結去抓留言
                
                if len(push_contents)>0:
                    try:
                        best_reply = calculate_worth_reply(push_contents, clean_title)
                        print("回答:", best_reply)

                        with open(f'{data_path}/PTT_Gossiping.txt','a+')as fp:
                            fp.write(clean_title+'  '+best_reply + '\n')
                    except:
                        continue
        index_value -= 1
       


    
    

   