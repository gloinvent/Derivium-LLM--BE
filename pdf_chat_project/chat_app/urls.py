from django.urls import path
from . import views

app_name = 'chat_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload_pdf/', views.upload_pdf, name='upload_pdf'),
    path('chat/', views.chat, name='chat'),
    path('summarize_chunk/', views.summarize_chunk, name='summarize_chunk'),
    path('check_status/<int:pdf_id>/', views.check_processing_status, name='check_processing_status'),
    path('chat_history/<int:pdf_id>/', views.get_chat_history, name='get_chat_history'),
    path('clear_data/', views.clear_data, name='clear_data'),
]