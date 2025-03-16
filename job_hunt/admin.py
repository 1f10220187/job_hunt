from django.contrib import admin
from .models import Company, UserProfile, University
# Register your models here.

# Company モデルの管理画面を設定
@admin.register(Company)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'matched')  # 一覧画面に表示する項目
    search_fields = ('name',)  # 検索フィールド

# UserProfile モデルの管理画面を設定
@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'want', 'mbti', 'pr')  # 一覧画面に表示する項目
    search_fields = ('user__username',)  # ユーザー名で検索できるように設定