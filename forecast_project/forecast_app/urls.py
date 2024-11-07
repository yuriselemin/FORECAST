from django.urls import path
from .views import index, analyze, download_csv

urlpatterns = [
    path('', index, name='index'),
    path('analyze/', analyze, name='analyze'),
    path('download/<int:stock_id>', download_csv, name='download_csv'),
]