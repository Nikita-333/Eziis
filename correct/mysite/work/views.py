import os

import numpy as np
from django.conf import settings
from django.http import FileResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse_lazy
from django.views.generic import DeleteView
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, BartForSequenceClassification, BartTokenizer, \
    pipeline
import torch
from work.forms import FileUploadForm
from work.models import FileProject
from django import template
from django.utils.html import mark_safe


# 9 вариант

class MetricCalculator:
    def __init__(self, results, file_contents, query):
        self.results = results
        self.file_contents = file_contents
        self.query = query

    def precision(self):
        # Расчет точности
        relevant_results = sum(1 for file, content in self.results if self.query in content)
        total_results = len(self.results)
        if total_results == 0:
            return 0.0
        precision = relevant_results / total_results
        return precision

    def recall(self):
        # Расчет полноты
        relevant_results = sum(1 for file, content in self.results if self.query in content)
        total_relevant = sum(1 for _, content in self.file_contents if self.query in content)
        if total_relevant == 0:
            return 0.0
        recall = relevant_results / total_relevant
        return recall

    def accuracy(self):
        # Расчет аккуратности
        true_positive = sum(1 for file, content in self.results if self.query in content)
        true_negative = len(self.file_contents) - len(self.results)
        total = len(self.file_contents)
        accuracy = (true_positive + true_negative) / total
        return accuracy

    def error(self):
        # Расчет ошибки
        return 1 - self.accuracy()

    def f_measure(self):
        # Расчет F-меры
        precision = self.precision()
        recall = self.recall()
        if precision + recall == 0:
            return 0.0
        f_measure = 2 * (precision * recall) / (precision + recall)
        return f_measure

    def trec_graph(self, num_points=11):
        precision_values = [i / num_points for i in range(num_points)]
        recall_values = [i / num_points for i in range(num_points)]

        # Построение графика
        plt.scatter(recall_values, precision_values, s=50, c='b', marker='o')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('TREC Precision-Recall Curve')
        plt.grid(True)
        # plt.savefig('trec_graph.png')  # Сохранение графика в файл

        return precision_values, recall_values

    def average_metrics(self):
        # Расчет усредненных метрик
        num_metrics = 5  # Число метрик, которые усредняются
        precision = self.precision()
        recall = self.recall()
        accuracy = self.accuracy()
        error = self.error()
        f_measure = self.f_measure()
        average_metrics = (precision + recall + accuracy + error + f_measure) / num_metrics
        return average_metrics

def search_results_metrics(request):
    query = request.GET.get('query', '')
    file_contents = []
    for file in FileProject.objects.all():
        file_path = os.path.join(settings.MEDIA_ROOT, str(file.file))
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            content = "File not found"
        file_contents.append((file, content))  # Store file and content as a tuple

    results = [(file, content) for file, content in file_contents if any(query in content for word in content.split())]

    metric_calculator = MetricCalculator(results, file_contents, query)  # Передаем query в MetricCalculator
    precision = metric_calculator.precision()
    recall = metric_calculator.recall()
    accuracy = metric_calculator.accuracy()
    error = metric_calculator.error()
    f_measure = metric_calculator.f_measure()
    average_metrics = metric_calculator.average_metrics()

    precision_values, recall_values = metric_calculator.trec_graph()

    return render(request, 'work/search_results_metrics.html', {
        'results': results,
        'query': query,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'error': error,
        'f_measure': f_measure,
        'average_metrics': average_metrics
    })


def index(request):
    return render(request, 'work/index.html')

def project_view(request):
    project = FileProject.objects.all()
    file_info = []
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    class_descriptions = ["politics", "sports", "technology", "games", "medicine", "programming", "education",
                          "environment", "food"]

    for file in project:
        file_path = os.path.join(settings.MEDIA_ROOT, str(file.file))
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            content = "File not found"

        result = classifier(content, class_descriptions)
        predicted_labels = [label for label, score in zip(result["labels"], result["scores"]) if score >= 0.1]

        file_info.append(
            {
                'id': file.id,
                'name': str(file.file),
                'content': content,
                'topic': predicted_labels,
            })

    return render(request, 'work/project.html', {'file_info': file_info})

def search_results(request):
    query = request.GET.get('query', '')
    file_contents = []
    for file in FileProject.objects.all():
        file_path = os.path.join(settings.MEDIA_ROOT, str(file.file))
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            content = "File not found"
        file_contents.append((file, content))  # Store file and content as a tuple

    vectorizer = TfidfVectorizer()#TfidfVectorizer используется для преобразования текста в
    # числовые векторы на основе метода TF-IDF (Term Frequency-Inverse Document Frequency).
    # TF-IDF позволяет оценить важность каждого слова в документе относительно всего корпуса текстов.
    file_vectors = vectorizer.fit_transform([content for _, content in file_contents])

    query_vector = vectorizer.transform([query])#Создается вектор запроса с помощью метода transform на экземпляре vectorizer. Вектор запроса представляет собой числовое представление запроса пользователя.

    similarities = file_vectors.dot(query_vector.T).toarray().flatten()#ычисляется схожесть между векторами файлов и вектором запроса с помощью операции умножения матрицы file_vectors на
    # транспонированный вектор запроса query_vector. Результат сохраняется в переменной similarities в виде массива чисел.
#Файлы сортируются по схожести с запросом, используя функцию sorted и zip. Мы создаем список кортежей, содержащих файлы и их схожесть с запросом, и сортируем его по второму элементу каждого кортежа (схожести) в порядке убывания.
    # Sort the files by similarity score
    sorted_files = [file for file, _ in sorted(zip(file_contents, similarities), key=lambda x: x[1], reverse=True)]

    results = [(file, content) for file, content in sorted_files if query in content]

    return render(request, 'work/search_results.html', {'results': results, 'query': query})


def upload_file(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save()
            return redirect('home')
    else:
        form = FileUploadForm()
    return render(request, 'work/index.html', {'form': form})


def delete_file(request, file_id):
    file_to_delete = FileProject.objects.get(pk=file_id)
    file_to_delete.file.delete()
    file_to_delete.delete()
    return redirect('home')


def download_file(request, file_id):
    file_object = get_object_or_404(FileProject, pk=file_id)
    response = FileResponse(file_object.file.open(), as_attachment=True)
    return response


def help_text(request):
    return render(request, 'work/help.html')
