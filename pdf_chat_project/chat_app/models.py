from django.db import models

class PDFDocument(models.Model):
    file = models.FileField(upload_to='pdfs/')
    filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    processing_time = models.FloatField(null=True, blank=True)
    num_pages = models.IntegerField(null=True, blank=True)
    num_chunks = models.IntegerField(null=True, blank=True)
    bm25_path = models.CharField(max_length=255, blank=True, null=True)
    tokenized_texts_path = models.CharField(max_length=255, blank=True, null=True)
    docs_path = models.CharField(max_length=255, blank=True, null=True)
    bm25_path = models.CharField(max_length=255, blank=True, null=True)
    tokenized_texts_path = models.CharField(max_length=255, blank=True, null=True)
    docs_path = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return self.filename

class ChatHistory(models.Model):
    pdf_document = models.ForeignKey(PDFDocument, on_delete=models.CASCADE)
    question = models.TextField()
    answer = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Question: {self.question[:50]} - PDF: {self.pdf_document.filename}"
