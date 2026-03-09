from django.contrib import admin
from django.contrib.contenttypes.admin import GenericTabularInline
from django.urls import reverse, path
from django.utils.html import format_html
from django.shortcuts import render
from .models import ConversationFile, RawTextDocument, HandwrittenNotesDocument, PDFDocument, VTTDocument, TeXDocument, TextChunk, Collection, CollectionPermission, WSConversation, GeminiAPIUsage, UserMemoryFact, EpisodicMemory
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


@admin.register(UserMemoryFact)
class UserMemoryFactAdmin(admin.ModelAdmin):
    list_display = ('user', 'category', 'fact_short', 'updated_at')
    list_filter = ('category', 'user')
    search_fields = ('fact',)
    ordering = ('user', 'category', 'created_at')

    def fact_short(self, obj):
        return (obj.fact or '')[:60] + ('…' if len(obj.fact or '') > 60 else '')
    fact_short.short_description = 'Fact'


@admin.register(EpisodicMemory)
class EpisodicMemoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'content_short', 'conversation', 'created_at')
    list_filter = ('user',)
    search_fields = ('content',)
    readonly_fields = ('user', 'content', 'conversation', 'assistant_message_uuid', 'created_at')

    def content_short(self, obj):
        return (obj.content or '')[:80] + ('…' if len(obj.content or '') > 80 else '')
    content_short.short_description = 'Content'

    def has_add_permission(self, request):
        return False  # Created automatically from conversation turns


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