from django import forms

from work.models import FileProject


class FileUploadForm(forms.ModelForm):
    class Meta:
        model = FileProject
        fields = ['file']
