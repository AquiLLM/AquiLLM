from django.contrib import admin
from django.contrib.contenttypes.admin import GenericTabularInline
from django.urls import reverse, path
from django.utils.html import format_html
from django.shortcuts import render
from django.contrib import messages as django_messages
from django.http import HttpResponseRedirect, FileResponse, Http404
from .models import ConversationFile, RawTextDocument, HandwrittenNotesDocument, PDFDocument, VTTDocument, TeXDocument, TextChunk, Collection, CollectionPermission, WSConversation, GeminiAPIUsage, FeedbackExport
from .ocr_utils import get_gemini_cost_stats


# class TextChunkInline(GenericTabularInline):
#     model = TextChunk
#     extra = 0
#     readonly_fields = ('start_position', 'end_position', 'chunk_number', 'admin_link')
#     can_delete = False
#     max_num = 0
#     fields = ('admin_link', 'start_position', 'end_position', 'chunk_number')

#     def has_add_permission(self, request, obj):
#         return False


#     def admin_link(self, instance):
#         url = reverse('admin:aquillm_textchunk_change', args=[instance.id])
#         return format_html('<a href="{}">View Details</a>', url)
#     admin_link.short_description = 'Details'


@admin.register(VTTDocument)
@admin.register(TeXDocument)
@admin.register(PDFDocument)
@admin.register(HandwrittenNotesDocument)
@admin.register(RawTextDocument)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'id')
    search_fields = ('title', 'full_text')
    # inlines = [TextChunkInline]


@admin.register(ConversationFile)
class ConversationFileAdmin(admin.ModelAdmin):
    list_display = ('name', 'conversation', 'id')
    search_fields = ('name',)
    list_filter = ('conversation',)



@admin.register(TextChunk)
class TextChunkAdmin(admin.ModelAdmin):
    list_display = ('chunk_number', 'start_position', 'end_position', 'document')
    search_fields = ('content',)

@admin.register(Collection)
class CollectionAdmin(admin.ModelAdmin):
    list_display = ('name',)
    search_fields = ('name',)

@admin.register(CollectionPermission)
class CollectionPermissionAdmin(admin.ModelAdmin):
    list_display = ('collection', 'user', 'permission')


@admin.register(WSConversation)
class WSConversationAdmin(admin.ModelAdmin):
    list_display = ('owner', 'id')


@admin.register(GeminiAPIUsage)
class GeminiAPIUsageAdmin(admin.ModelAdmin):
    list_display = ('operation_type', 'timestamp', 'input_tokens', 'output_tokens', 'cost')
    list_filter = ('operation_type', 'timestamp')
    date_hierarchy = 'timestamp'
    readonly_fields = ('timestamp', 'operation_type', 'input_tokens', 'output_tokens', 'cost')

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    # Custom admin view for cost summary
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('summary/', self.admin_site.admin_view(self.cost_summary_view), name='gemini-cost-summary'),
        ]
        return custom_urls + urls

    def cost_summary_view(self, request):
        stats = get_gemini_cost_stats()
        context = {
            'title': 'Gemini API Cost Summary',
            'stats': stats,
            'opts': self.model._meta,
            **self.admin_site.each_context(request),
        }
        return render(request, 'aquillm/gemini_cost_monitor.html', context)


@admin.register(FeedbackExport)
class FeedbackExportAdmin(admin.ModelAdmin):
    list_display = ('id', 'status', 'created_at', 'completed_at', 'triggered_by',
                    'conversation_count', 'message_count', 'download_link')
    list_filter = ('status',)
    readonly_fields = ('status', 'created_at', 'completed_at', 'triggered_by', 'file',
                       'conversation_count', 'message_count', 'error_message', 'celery_task_id')

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def download_link(self, obj):
        if obj.status == 'completed' and obj.file:
            url = reverse('admin:feedback-export-download', args=[obj.pk])
            return format_html('<a href="{}">Download JSON</a>', url)
        return '-'
    download_link.short_description = 'Download'

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('trigger/', self.admin_site.admin_view(self.trigger_export_view),
                 name='feedback-export-trigger'),
            path('<int:export_id>/download/', self.admin_site.admin_view(self.download_export_view),
                 name='feedback-export-download'),
        ]
        return custom_urls + urls

    def download_export_view(self, request, export_id):
        try:
            export = FeedbackExport.objects.get(pk=export_id)
        except FeedbackExport.DoesNotExist:
            raise Http404
        if not export.file:
            raise Http404
        return FileResponse(export.file.open('rb'), content_type='application/json',
                            as_attachment=True, filename=export.file.name.split('/')[-1])

    def trigger_export_view(self, request):
        if request.method == 'POST':
            export = FeedbackExport.objects.create(triggered_by=request.user)
            from .feedback_tasks import generate_feedback_export
            generate_feedback_export.delay(export.pk)
            django_messages.success(request, f'Feedback export #{export.pk} has been queued.')
            return HttpResponseRedirect(reverse('admin:aquillm_feedbackexport_changelist'))
        return render(request, 'aquillm/feedback_export_trigger.html', {
            'title': 'Generate Feedback Export',
            'opts': self.model._meta,
            **self.admin_site.each_context(request),
        })

    def changelist_view(self, request, extra_context=None):
        extra_context = extra_context or {}
        extra_context['trigger_url'] = reverse('admin:feedback-export-trigger')
        return super().changelist_view(request, extra_context=extra_context)