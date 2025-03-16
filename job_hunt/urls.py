from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from django.contrib.auth.decorators import login_required

urlpatterns = [
    path('', login_required(views.index), name='index'),
    path('iniad/',views.iniad_vector,name='iniad'),
    path('detail<int:pk>/',views.detail,name='detail'),
    path('matched<int:pk>/',views.generate_matched,name="matched"),
    path('pr<int:pk>/',views.generate_pr_check,name="pr_check"),
    path('effort<int:pk>/',views.generate_effort_check,name="effort_check"),
    path('pr_correction/',views.pr_correction,name="pr_correction"),
    path('effort_correction/',views.effort_correction,name="effort_correction"),
    path('delete<int:pk>/',views.delete,name="delete"),
    path('accounts/login/',auth_views.LoginView.as_view(),name="login"),
    path('accounts/logout/',views.logout_views,name="logout"),
    path('accounts/signup/',views.signup,name="signup"),
    path('accounts/profile/',login_required(views.profile),name="profile"),
]