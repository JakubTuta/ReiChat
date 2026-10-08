from rest_framework import serializers

from . import models


class AIModelVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.AIModelVersion
        fields: list[str] = ["parameters", "size", "size_bytes", "verified_at"]


class AIModelSerializer(serializers.ModelSerializer):
    versions = AIModelVersionSerializer(many=True, read_only=True)

    class Meta:
        model = models.AIModel
        fields: list[str] = [
            "id",
            "name",
            "model",
            "description",
            "popularity",
            "can_process_image",
            "is_embedding",
            "can_use_tools",
            "versions",
            "index",
        ]


class PersonaSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Persona
        fields: list[str] = ["id", "name", "system_prompt", "created_at"]


class PromptTemplateSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.PromptTemplate
        fields: list[str] = ["id", "name", "description", "content", "created_at"]


class MCPServerSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.MCPServer
        fields: list[str] = ["id", "name", "transport", "command", "url", "enabled", "auto_approve", "created_at"]


class DocumentCollectionSerializer(serializers.ModelSerializer):
    embedding_model_name = serializers.CharField(source="embedding_model.model", read_only=True)
    document_count = serializers.IntegerField(read_only=True, default=0)

    class Meta:
        model = models.DocumentCollection
        fields: list[str] = [
            "id", "name", "chat", "embedding_model", "embedding_model_name",
            "embedding_parameters", "document_count", "created_at",
        ]


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Document
        fields: list[str] = [
            "id", "collection", "filename", "status", "error_message", "chunk_count", "uploaded_at",
        ]
        read_only_fields: list[str] = ["status", "error_message", "chunk_count"]


class ChatHistorySerializer(serializers.ModelSerializer):
    ai_model = AIModelSerializer(read_only=True)

    class Meta:
        model = models.ChatHistory
        fields = [
            "id",
            "ai_model",
            "title",
            "last_update_time",
        ]

    def create(self, validated_data):
        ai_model = self.context.get("ai_model")

        validated_data["ai_model"] = ai_model

        return models.ChatHistory.objects.create(**validated_data)
