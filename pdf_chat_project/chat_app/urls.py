from django.urls import path
from . import views

app_name = 'chat_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload_pdf/', views.upload_pdf, name='upload_pdf'),
    path('chat/', views.chat, name='chat'),
    path('history/<int:pdf_id>/', views.get_chat_history, name='get_chat_history'),
    path('summarize_chunk/', views.summarize_chunk, name='summarize_chunk'),
    path('clear_data/', views.clear_data, name='clear_data'),
]