from django.contrib.auth.models import User
from django.db import models
# Create your models here.

class UserProfile(models.Model):
    MBTI_CHOICES = [
        ('INTJ', 'INTJ'), ('INTP', 'INTP'), ('ENTJ', 'ENTJ'), ('ENTP', 'ENTP'),
        ('INFJ', 'INFJ'), ('INFP', 'INFP'), ('ENFJ', 'ENFJ'), ('ENFP', 'ENFP'),
        ('ISTJ', 'ISTJ'), ('ISFJ', 'ISFJ'), ('ESTJ', 'ESTJ'), ('ESFJ', 'ESFJ'),
        ('ISTP', 'ISTP'), ('ISFP', 'ISFP'), ('ESTP', 'ESTP'), ('ESFP', 'ESFP')
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")  # ユーザーと1対1で紐付け
    want = models.TextField(null=True, blank=True)
    avoid = models.TextField(null=True, blank=True)
    career = models.TextField(null=True, blank=True)
    subjects=models.JSONField(default=list,null=True, blank=True)
    skill = models.TextField(null=True, blank=True)
    mbti = models.CharField(max_length=4, choices=MBTI_CHOICES,null=True, blank=True) # NullでもOKにする
    pr = models.TextField(null=True,blank=True)
    pr_check = models.TextField(null=True,blank=True)
    effort = models.TextField(null=True, blank=True)
    effort_check = models.TextField(null=True,blank=True)
    def __str__(self):
        return f"{self.user.username} Profile (志望業界: {self.want}, 科目群: {self.subjects})"

class University(models.Model):
    university_vector_path = models.TextField()
    

class Company(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)  # 企業名
    vectorstore_path = models.TextField()  # 企業情報のベクトルストア
    description = models.TextField(null=True, blank=True)  # 説明や備考
    matched = models.TextField(null=True, blank=True)
    pr_check = models.TextField(null=True,blank=True)
    effort_check = models.TextField(null=True,blank=True)
    def __str__(self):
        return self.name


class ChatLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # チャットを行ったユーザー
    company = models.ForeignKey('Company', on_delete=models.CASCADE, null=True, blank=True, related_name="chat_logs")  # 関連する企業
    question = models.TextField()  # ユーザーの質問
    answer = models.TextField()    # アプリの応答
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.created_at}"
