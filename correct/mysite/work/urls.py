from django.urls import path, re_path

from .views import *

urlpatterns = [
    path('', index, name='home'),
    path('project_print/', project_view, name='project_view'),
    path('search/', search_results, name='search_results'),
    path('help/',help_text,name='help_text'),
    path('upload/', upload_file, name='upload_file'),
    path('metrics/',search_results_metrics,name='search_results_metrics'),
    path('delete/<int:file_id>/', delete_file, name='delete_file'),
    path('download_file/<int:file_id>/', download_file, name='download_file'),
]