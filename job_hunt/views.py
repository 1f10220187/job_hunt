from django.shortcuts import render,redirect,get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.views import LoginView
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
import os
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
import io
import fitz  # PyMuPD
from langchain.schema import Document
import hashlib
from bs4 import BeautifulSoup, SoupStrainer
from django.conf import settings
import time
import urllib.robotparser
from urllib.parse import urljoin, urlparse
from .models import University, Company, UserProfile, ChatLog
from .forms import SignUpForm
from django.core.validators import URLValidator
from django.core.exceptions import ValidationError
import re
import shutil
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.schema import SystemMessage
import traceback

#環境変数呼び出し
openai_api_key = settings.OPENAI_API_KEY
langchain_api_key = settings.LANGCHAIN_API_KEY
openai_api_base = settings.OPENAI_API_BASE

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_base=openai_api_base)

persist_directory = "./vectorstore_data/iniad" #作成したvectorstoreの保存先
embeddings = OpenAIEmbeddings(openai_api_base=openai_api_base)

###################プロンプトエリア##############################
default_prompt = """あなたはとても優秀な就職活動のアシスタントです！
    質問に答えるために、検索された文脈の以下の部分を使用してください。
    答えがわからない場合は、わからないと答えましょう。
    回答は長すぎてはいけませんがなるべく親身に回答してあげましょう。
    """

pr_prompt = """あなたは優秀な就活エージェントです。  
以下の基準をもとにユーザーの自己PRを添削し、評価してください。  
また、企業に対する印象を向上させるための改善点やアドバイスを提供してください。

以下は「良い自己PR」の要件の例です。参考にして添削を行いましょう：
- **結論を先に述べる**: ユーザーの強みやアピールポイントを簡潔に冒頭で提示する。
- **具体的なエピソードで裏付ける**: 強みを証明するエピソードを明確に説明する。
- **企業での活かし方を示す**: ユーザーのスキルや経験が応募先企業でどのように活かせるかを具体的に示す。
- **簡潔で読みやすい表現**: 冗長な表現を避け、読み手にストレスを与えない文章にする。

出力形式:
1. 自己PRに対する全体的な評価 (例: A～C評価)
2. 評価の根拠
3. 改善点の具体的なアドバイス (箇条書き形式)
"""

effort_prompt = """あなたは優秀な就活エージェントです。
以下の基準をもとにユーザーのガクチカ（学生時代に力を入れたこと）を添削し、評価してください

以下は「良いガクチカ」の要件の例です。参考にして添削を行いましょう：
1. **具体的なエピソード**：
   - 実際に取り組んだ活動や経験が具体的に述べられているか。
   - どのような活動をして、どの役割を担ったかが明確か。

2. **行動と結果の明確さ**：
   - どのような行動を取ったか、その結果としてどのような成果を得たかが具体的に示されているか。
   - 定量的な成果や具体的な成果があれば、それが記載されているか。

3. **企業での活かし方**：
   - この経験から得たスキルや知識が、応募先企業でどのように活かせるかが示されているか。

4. **一貫性**：
   - 自己PRや志望動機と整合性が取れているか。
   - 自分の強みや価値観がどのように反映されているか。

5. **簡潔で読みやすい表現**：
   - 文が簡潔で、要点が明確に伝わるか。
   - 箇条書きや段落分けが適切で、採用担当者が読みやすいか。

出力形式:
1. ガクチカに対する全体的な評価 (例: A～C評価)
2. 評価の根拠
3. 改善点の具体的なアドバイス (箇条書き形式)
"""


company_pr_prompt = """
あなたは優秀な就活エージェントです。
質問に答えるために、検索された文脈の以下の部分を使用してください。

出力形式:
1. 自己PRに対する全体的な評価 (例: A～C評価)
2. 高く評価されそうなポイント
3. あまり評価されなそうなポイント
4. 改善点の具体的なアドバイス (箇条書き形式)
"""

company_effort_prompt = """
あなたは優秀な就活エージェントです。
質問に答えるために、検索された文脈の以下の部分を使用してください。

出力形式:
1. ガクチカに対する全体的な評価 (例: A～C評価)
2. 高く評価されそうなポイント
3. あまり評価されなそうなポイント
4. 改善点の具体的なアドバイス (箇条書き形式)
"""

def make_detail_prompt(request):
    user = request.user
    user_profile = UserProfile.objects.get(user=user)
    want = user_profile.want or "未記入"
    avoid = user_profile.avoid or "未記入"
    career = user_profile.career or "未記入"
    subjects = ", ".join(user_profile.subjects) if user_profile.subjects else "未記入"
    skill = user_profile.skill or "未記入"
    mbti = user_profile.mbti or "未記入"

    prompt = f"""
    あなたはとても優秀な就職活動のアシスタントです！
    質問に答えるために、検索された文脈の以下の部分を使用してください。
    答えがわからない場合は、わからないと答えましょう。
    回答は長すぎてはいけませんがなるべく親身に回答してあげましょう。
    以下に質問しているユーザーの情報を記載しているので必要な場合にのみ参照するようにしてください。

    [ユーザー情報]
    - 所属大学: INIAD
    - 会社選びの軸: {want}
    - 避けたい仕事内容や環境: {avoid}
    - 歩みたいキャリアプラン: {career}
    - 科目群: {subjects}
    - スキル: {skill}
    - MBTI: {mbti}
    """
    return prompt.strip()

def make_matched_prompt(request):
    user = request.user
    user_profile = UserProfile.objects.get(user=user)
    want = user_profile.want or "未記入"
    avoid = user_profile.avoid or "未記入"
    career = user_profile.career or "未記入"
    skill = user_profile.skill or "未記入"
    mbti = user_profile.mbti or "未記入"

    prompt = f"""
    あなたはとても優秀な企業と学生の相性を判断するAIです。
    質問に答えるために、検索された文脈を使用してください。
    ユーザーは就活生です、自分に合った企業で働けるようにサポートしてあげましょう！
    以下は質問しているユーザーの情報です。

    [ユーザー情報]
    - 所属大学: INIAD
    - 企業選びで大切にしていること: {want}
    - 避けたい仕事内容や環境: {avoid}
    - 歩みたいキャリアプラン: {career}
    - スキル: {skill}
    - MBTI: {mbti}
    """
    return prompt.strip()

###################################################################
# Create your views here.


@login_required
def index(request):
    user = request.user
    username = user.username
    univ = University.objects.first()
    if request.method=="POST":
        name = request.POST.get('name')
        web_url = request.POST.get('web_url')
        pdf_url = request.POST.get('pdf_url')

        # URLバリデーション
        validator = URLValidator()
        if web_url:
            try:
                validator(web_url)
            except ValidationError:
                return render(request, "job_hunt/index.html", {
                    'error_message': "無効なWeb URLです。"
                })
        
        if pdf_url:
            try:
                validator(pdf_url)
            except ValidationError:
                return render(request, "job_hunt/index.html", {
                    'error_message': "無効なPDF URLです。"
                })

        if not web_url and not pdf_url:
            return render(request,"job_hunt/index.html",{
                'error_message':"web_urlかpdf_urlのどちらかは入力してください"
            })
        
        try:
            # univ_vector_path=os.path.join(settings.VECTORSTORE_DIR, 'iniad')
            univ_vector_path = univ.university_vector_path
            print(f"{name}のベクトルストア作成を開始します")
            vectorstore,vectorstore_dir =append_to_vectorstore(univ_vector_path,username,name,web_url=web_url,
                                                                pdf_url=pdf_url,max_pages=25)
            if vectorstore is None:
                return "同じ名前のベクトルストアが存在している可能性があります"
            rag_chain = create_rag_chain(vectorstore)
            summary = rag_chain.invoke('こちらの企業の業界や事業内容やについてわかりやすく丁寧に教えてください')
            Company.objects.create(
                user=request.user,
                name=name,
                vectorstore_path =vectorstore_dir,
                description = summary,
                matched = None,
                pr_check = None,
            )
            return redirect('index')
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            return render(request,'job_hunt/index.html',{
                'error_message':f"ベクトルストア作成中にエラーが発生しました{e}"
            })
    company_list = Company.objects.filter(user=request.user)
    return render(request, 'job_hunt/index.html',{"company_list":company_list})

@login_required
def detail(request,pk):
    company = get_object_or_404(Company,pk=pk)
    chat_logs = company.chat_logs.all()
    company_vectorstore_path = company.vectorstore_path

    # ベクトルストアをロード
    try:
        company_vectorstore = Chroma(persist_directory=company_vectorstore_path, embedding_function=embeddings)

        if request.method == 'POST':
            question = request.POST.get('question')
            check = request.POST.get('include_user_info')
            if check:
                prompt = make_detail_prompt(request)
                rag_chain = create_rag_chain(company_vectorstore,prompt=prompt)
                answer = rag_chain.invoke(question)
            else:
                rag_chain = create_rag_chain(company_vectorstore)
                answer = rag_chain.invoke(question)

            chat = ChatLog.objects.create(
                user=request.user,
                company=company,
                question=question,
                answer=answer,
            )
            return JsonResponse({
            'question': chat.question,
            'answer': chat.answer,
            })
            # return redirect('detail',pk=company.pk)
    except:
        print(traceback.format_exc())
        return JsonResponse({'error': '内部エラーが発生しました'}, status=500)
    return render(request, "job_hunt/detail.html", {"company": company,"chat_logs":chat_logs})

@login_required
def generate_matched(request,pk):
    company = get_object_or_404(Company,pk=pk)
    name=company.name
    question = f"""ユーザーとこの企業({name})の相性を診断し、次の形式で答えてください：
                1. **相性が良いポイント**: ユーザーと企業が特にマッチする点を推測し教えてください。\n
                2. **ミスマッチの可能性**: ユーザーと企業で課題やミスマッチが発生しそうな点を推測し教えてください。\n
                3. **ユーザーのMBTIが記入されている場合は、求める人物像などと一致しているか教えてください**\n

                4. 最終的にマッチとミスマッチどちらが起こる可能性が高いのかを理由とともに教えてください。\n
                """
    company_vectorstore_path = company.vectorstore_path
    company_vectorstore = Chroma(persist_directory=company_vectorstore_path, embedding_function=embeddings)
    if request.method == 'POST':
        prompt = make_matched_prompt(request)
        rag_chain = create_rag_chain(company_vectorstore,prompt=prompt)
        answer = rag_chain.invoke(question)
        company.matched = answer
        company.save()
        print(request, "相性診断を完了しました！")
    return redirect('detail',pk=pk)

@login_required
def generate_pr_check(request,pk):
    user = request.user
    user_profile = UserProfile.objects.get(user=user)
    company = get_object_or_404(Company,pk=pk)
    pr = user_profile.pr or "未記入"
    company_name = company.name
    question = f"""
    以下の自己PRがこの企業({company_name})にどのように評価されるかを分析し、求める人物像や、実際の業務内容などから、この企業の視点で具体的な評価を出してください。また、改善点があれば簡潔に教えてください。
    {pr}
    """
    company_vectorstore_path = company.vectorstore_path
    company_vectorstore = Chroma(persist_directory=company_vectorstore_path, embedding_function=embeddings)
    if request.method == 'POST':
        rag_chain = create_rag_chain(company_vectorstore,prompt=company_pr_prompt)
        answer = rag_chain.invoke(question)
        company.pr_check = answer
        company.save()
        print(request, "自己prチェックを完了しました！")
    return redirect('detail',pk=pk)

@login_required
def generate_effort_check(request,pk):
    user = request.user
    user_profile = UserProfile.objects.get(user=user)
    company = get_object_or_404(Company,pk=pk)
    effort = user_profile.effort or "未記入"
    company_name = company.name
    question = f"""
    以下のガクチカがこの企業({company_name})にどのように評価されるかを分析し、求める人物像や、実際の業務内容などから、この企業の視点で具体的な評価を出してください。また、改善点があれば簡潔に教えてください。
    {effort}
    """
    company_vectorstore_path = company.vectorstore_path
    company_vectorstore = Chroma(persist_directory=company_vectorstore_path, embedding_function=embeddings)
    if request.method == 'POST':
        rag_chain = create_rag_chain(company_vectorstore,prompt=company_effort_prompt)
        answer = rag_chain.invoke(question)
        company.effort_check = answer
        company.save()
        print(request, "ガクチカチェックを完了しました！")
    return redirect('detail',pk=pk)

@login_required
def pr_correction(request):
    univ = University.objects.first()
    univ_vector_path = univ.university_vector_path
    univ_vector = Chroma(persist_directory=univ_vector_path, embedding_function=embeddings)
    user = request.user
    user_profile = UserProfile.objects.get(user=user)
    pr = user_profile.pr or "未記入"
    effort_check = user_profile.effort_check or "まだ添削してないよ"
    pr_check = user_profile.pr_check or "まだ添削してないよ"
    question = f"""
    以下の自己PRを評価してください。
    {pr}
    """
    if request.method == 'POST':
        rag_chain = create_rag_chain(univ_vector,prompt=pr_prompt)
        pr_check = rag_chain.invoke(question)
        user_profile.pr_check = pr_check
        user_profile.save()
        print("自己prの添削完了")
    return render(request,"job_hunt/correction.html",{"pr_check":pr_check,"effort_check":effort_check})


@login_required
def effort_correction(request):
    univ = University.objects.first()
    univ_vector_path = univ.university_vector_path
    univ_vector = Chroma(persist_directory=univ_vector_path, embedding_function=embeddings)
    user = request.user
    user_profile = UserProfile.objects.get(user=user)
    effort=user_profile.effort or "未記入"
    effort_check = user_profile.effort_check or "まだ添削してないよ"
    question = f"""
    以下のガクチカを添削してください
    {effort}
    """
    if request.method=='POST':
        rag_chain = create_rag_chain(univ_vector,prompt=effort_prompt)
        effort_check = rag_chain.invoke(question)
        user_profile.effort_check = effort_check
        user_profile.save()
        print("ガクチカ添削完了")
    return redirect('pr_correction')

@login_required
def delete(request,pk):
    company = get_object_or_404(Company,pk=pk)
    if company.vectorstore_path:
        shutil.rmtree(company.vectorstore_path, ignore_errors=True)
    company.delete()
    return redirect('index')


#########################INIADのベクトルストアを作る#################################################################

def iniad_vector(request):
    web_url = "https://www.iniad.org/iniad-concept/"
    pdf_url = "https://static-files.iniad.org/sites/1/2024/08/INIAD30_ja_small.pdf"
    try:
        # 既存のデータを削除
        existing_universities = University.objects.all()
        if existing_universities.exists():
            existing_universities.delete()
            print("既存のUniversityレコードを削除しました。")

        # 新しいベクトルストアを作成
        vectorstore,vectorstore_dir = create_vectorstore(web_url,pdf_url, 'iniad',15)
        if vectorstore is None or vectorstore_dir is None:
            raise ValueError("INIADベクトルストアの作成に失敗しました")
        University.objects.create(
            university_vector_path=vectorstore_dir
        )
    except Exception as e:
        print(f"iniadのベクトルストア作成中にエラーが発生しました: {e}")
        return HttpResponse(f"エラー: {e}", status=500)
    print("できた")
    return HttpResponse("INIADのベクトルストア作成が完了しました！")
#######################################################################################################################

###################ログイン関係#########################################################################################

#ログイン
class CustomLoginView(LoginView):
    template_name = "registration/login.html"

#ログアウト
def logout_views(request):
    logout(request)
    return redirect('index')

#サインアップ
def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form=SignUpForm()
    return render(request,"registration/signup.html",{"form":form})


#プロフィール
@login_required
def profile(request):
    subjects_list = [
        ('システム科目群', 'System'),
        ('ソフトウェア科目群', 'Software'),
        ('データサイエンス科目群', 'ds'),
        ('ユーザー・エクスペリエンス科目群', 'ux'),
        ('ICT社会応用科目群', 'ict'),
        ('ビジネス構築科目群', 'business'),
        ('コミュニティ形成科目群', 'comunity')
    ]

    mbti_list = ["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP", "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"]

    user=request.user
    user_profile, created = UserProfile.objects.get_or_create(user=user)

    if request.method == 'POST':
        want = request.POST.get('want', user_profile.want)  # デフォルト値: user_profile.want
        avoid = request.POST.get('avoid', user_profile.avoid)
        career = request.POST.get('career', user_profile.career)
        subjects = request.POST.getlist('Subject', user_profile.subjects)
        skill = request.POST.get('skill',user_profile.skill)
        mbti = request.POST.get('mbti', user_profile.mbti)  # デフォルト値: user_profile.mbti
        pr = request.POST.get('pr',user_profile.pr)
        effort =request.POST.get('effort',user_profile.effort)

        if user_profile:
            user_profile.want = want
            user_profile.avoid = avoid
            user_profile.career = career
            user_profile.subjects = subjects
            user_profile.skill = skill
            user_profile.mbti = mbti
            user_profile.pr = pr
            user_profile.effort = effort        
        user_profile.save()

        # 更新後にリロードする（リダイレクトでPOSTの再送信を防ぐ）
        return redirect('profile')

    context = {
        "user":user,
        "subjects_list":subjects_list,
        "mbti_list":mbti_list,
        "want":user_profile.want or "",
        "avoid":user_profile.avoid or "",
        "career":user_profile.career or "",
        "subjects":user_profile.subjects or[],
        "skill":user_profile.skill or "",
        "mbti":user_profile.mbti or "",
        "pr":user_profile.pr or "",
        "effort":user_profile.effort or "",
    }

    return render(request,"accounts/profile.html",context)

#######################################################################################################################



######################RAGで使う関数#####################################################################################
def get_pdf(pdf_url):
    print(f"pdf読み込み中{pdf_url}")
    response = requests.get(pdf_url)
    pdf_file = io.BytesIO(response.content)

    # PDFを開く
    pdf_document = fitz.open(stream=pdf_file, filetype="pdf")

    result = ""
    # 各ページを読み込んでテキストを取得
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text = page.get_text()
        result += text

    # PDFを閉じる
    pdf_document.close()

    return result


def get_webpage(start_url, max_pages, delay=3):
    visited_urls = set()  # 訪問済みURL
    to_visit_urls = {start_url}  # 訪問予定URL
    all_texts = []  # 取得したテキストを格納するリスト

    # robots.txt を確認
    parsed_url = requests.utils.urlparse(start_url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        print("Robots.txtを確認しました")
    except Exception as e:
        print(f"robots.txtの読み取り中にエラーが発生しました: {e}")
        delay = 6
        print(f"robots.txtがないため、リクエスト間隔を{delay}秒に設定します。")

    while to_visit_urls and len(visited_urls) < max_pages:
        print(f"webページ探索中{len(visited_urls)}")
        url = to_visit_urls.pop()
        if not rp.can_fetch("*", url):  # robots.txtに従う
            print(f"robots.txtによってアクセスが禁止されています: {url}")
            time.sleep(delay)
            continue

        try:
            visited_urls.add(url)
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # ステータスコードを確認

            # ページの本文部分を解析
            soup = BeautifulSoup(response.content, "html.parser", parse_only=SoupStrainer("body"))
            loader = WebBaseLoader(web_paths=(url,), bs_kwargs=dict(parse_only=SoupStrainer("body")))
            page_docs = loader.load()
            all_texts.extend(page_docs)

            # ページ内のリンクを収集
            for link in soup.find_all("a", href=True):
                href = link["href"]

                try:
                    # 相対URLを絶対URLに変換
                    full_url = urljoin(start_url, href)

                    # 同じドメイン内のURLのみ収集
                    if urlparse(full_url).netloc == urlparse(start_url).netloc:
                        if full_url not in visited_urls and full_url not in to_visit_urls:
                            to_visit_urls.add(full_url)
                except Exception as e:
                    print(f"リンク解析中にエラーが発生しました: {e}")
            print(f"取得成功: {url}")
            time.sleep(delay)  # リクエスト間隔を設定

        except requests.exceptions.RequestException as e:
            print(f"リクエスト中にエラーが発生しました: {e}")
            time.sleep(delay)

    return all_texts


def create_vectorstore(web_url,pdf_url, vectorstore_name,max_pages):
    
    try:
        print("ベクトルストア作成中")
        vectorstore_dir = os.path.join(settings.VECTORSTORE_DIR, vectorstore_name)
        if os.path.exists(vectorstore_dir):
            print(f"ベクトルストア{vectorstore_name}を上書きします")
        if not os.path.exists(vectorstore_dir):
            os.makedirs(vectorstore_dir)
        print(vectorstore_dir)

        # テキストを取得
        if pdf_url == None:
            result =get_webpage(web_url,max_pages)
            docs = [Document(page_content=result, metadata={})]
        elif web_url == None:
            docs = get_pdf(pdf_url)
        else:
            result_web=get_webpage(web_url,max_pages)
            pdf_text=get_pdf(pdf_url)
            result_pdf = [Document(page_content=pdf_text, metadata={})]

            docs = result_web+result_pdf
        if not docs:
            raise ValueError("テキストを取得できませんでした。")

        # テキスト分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # ベクトルストアの作成
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=vectorstore_dir)
        return vectorstore,vectorstore_dir

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None,None
    
def append_to_vectorstore(univ_vector_path,username,vector_name,web_url=None,pdf_url=None,max_pages=10):
    print(f"web_url={web_url}, pdf_url={pdf_url}")
    try:
        print("ベクトルストアのコピーと追記を開始します")
        # 1️⃣ 新しいベクトルストアの保存先
        vectorstore_dir = os.path.join(settings.VECTORSTORE_DIR, username,vector_name)

        # すでに同名のベクトルストアが存在する場合はエラーを出す
        if os.path.exists(vectorstore_dir):
            print(f"{vectorstore_dir} はすでに存在します！上書きします。")
            shutil.rmtree(vectorstore_dir)  # 既存のベクトルストアを削除
        
        if os.path.exists(univ_vector_path):
            print(f"大学のベクトルストア {univ_vector_path} を {vectorstore_dir} にコピー中...")
            shutil.copytree(univ_vector_path, vectorstore_dir)
            print(f"コピーが完了しました！")
        else:
            print("大学のベクトルストアを作成")
            iniad_vector()
            print(f"大学のベクトルストア {univ_vector_path} を {vectorstore_dir} にコピー中...")
            shutil.copytree(univ_vector_path, vectorstore_dir)
            print(f"コピーが完了しました！")

 
        # 3️⃣ 企業の新しいベクトルストアをロード
        vectorstore = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)

        # 4️⃣ 追加するドキュメントを作成
        if not pdf_url and not web_url:
            raise ValueError("web_urlかpdf_urlのどちらかは必須です")

        # テキストを取得
        if pdf_url is None or pdf_url == "":
            print("appendでif pdf is Noneがよばれたよ")
            docs =get_webpage(web_url,max_pages)
            # docs = [Document(page_content=result, metadata={})]
        elif web_url is None or web_url == "":
            print("appendでif web is Noneがよばれたよ")
            docs = get_pdf(pdf_url)
        else:
            print("appendでelseがよばれたよ")
            result_web=get_webpage(web_url,max_pages)
            pdf_text=get_pdf(pdf_url)
            result_pdf = [Document(page_content=pdf_text, metadata={})]

            docs = result_web+result_pdf
        if not docs:
            raise ValueError("テキストを取得できませんでした。")
        
        vectorstore.add_documents(docs)

        print(f"ベクトルストアへの追記が完了しました！")
        return vectorstore, vectorstore_dir
    except Exception as e:
        print(f"追記中にエラーが発生しました: {e}")
        return None, None

def format_docs(docs_list):
    return "\n\n".join(doc.page_content for doc in docs_list)

def create_rag_chain(vectorstore,prompt=default_prompt):
    print('rag_chain作成中')
    retriever = vectorstore.as_retriever()
    #prompt = hub.pull("rlm/rag-prompt")
    prompt = prompt + """
    文脈:
    {context}

    質問:
    {question}"""
    # System メッセージとして設定
    system_message = SystemMessage(content=prompt)
    # Human メッセージをテンプレートに
    human_template = "{question}"
    human_message = HumanMessagePromptTemplate.from_template(human_template)
    # prompt_template = PromptTemplate.from_template(prompt)
    # PromptTemplate を生成
    chat_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(system_message.content), human_message])
    print(chat_prompt)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | chat_prompt
        | llm
        | StrOutputParser()
    )


###################################################################################################################################