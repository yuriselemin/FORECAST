from django import forms

class AnalyzeForm(forms.Form):
    ticker = forms.ChoiceField(label="Акция")
    start_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}), label="Начало периода")
    end_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}), label="Конец периода")
    model_type = forms.ChoiceField(choices=[
        ('sklearn', 'Scikit-learn'),
        ('tensorflow', 'TensorFlow'),
        ('pytorch', 'PyTorch'),
    ], label="Тип модели")

    def __init__(self, ticker_choices, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['ticker'].choices = ticker_choices