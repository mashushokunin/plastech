from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "meeting_report_2026-10-20_final.pdf"
SLATE = ROOT / "outputs" / "role_based_design" / "experiment_candidates_by_profile.csv"
ERRORS = ROOT / "config" / "prediction_error_reference.json"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2E6F95")
PALE_BLUE = colors.HexColor("#EAF3F8")
PALE_GREEN = colors.HexColor("#EAF5EF")
PALE_YELLOW = colors.HexColor("#FFF4D6")
RED = colors.HexColor("#A33A2B")
GREY = colors.HexColor("#5F6B73")
LIGHT_GREY = colors.HexColor("#F3F5F6")


def register_fonts() -> None:
    asset_root = Path("/System/Library/AssetsV2")
    regular = next(asset_root.glob("**/BIZ_UDMincho-regular.ttf"))
    # 同じ日本語TrueTypeフォントを別名登録し、太字役はサイズと色で階層化する。
    # macOS標準の一部CJKフォントはReportLabのsubset保存で失敗するため使用しない。
    bold = regular
    pdfmetrics.registerFont(TTFont("JP", str(regular)))
    pdfmetrics.registerFont(TTFont("JP-Bold", str(bold)))


def paragraph(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(str(text), style)


def styled_table(data, widths, *, header=True, font_size=8.0, align="LEFT") -> Table:
    rows = []
    for row_index, row in enumerate(data):
        style = STYLES["table_header"] if header and row_index == 0 else STYLES["table_cell"]
        rows.append([paragraph(value, style) for value in row])
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign=align)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD3D8")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ]
        )
        for index in range(1, len(rows)):
            if index % 2 == 0:
                commands.append(("BACKGROUND", (0, index), (-1, index), LIGHT_GREY))
    table.setStyle(TableStyle(commands))
    return table


def card(title: str, body: str, color=PALE_BLUE) -> Table:
    content = [
        paragraph(title, STYLES["card_title"]),
        Spacer(1, 2 * mm),
        paragraph(body, STYLES["body"]),
    ]
    table = Table([[content]], colWidths=[174 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color),
                ("BOX", (0, 0), (-1, -1), 0.6, BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def bullet(text: str) -> Paragraph:
    return Paragraph(f"• {text}", STYLES["bullet"])


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D6DDE1"))
    canvas.line(18 * mm, 13 * mm, 192 * mm, 13 * mm)
    canvas.setFont("JP", 7)
    canvas.setFillColor(GREY)
    canvas.drawString(18 * mm, 8 * mm, "プラス・テク共同検討 | 2026-10-20")
    canvas.drawRightString(192 * mm, 8 * mm, f"{doc.page}")
    canvas.restoreState()


def profile_table(frame: pd.DataFrame, profile: str) -> Table:
    selected = frame[frame["selection_profile"] == profile].sort_values("profile_rank")
    data = [["候補", "PVC系列", "材料数", "硬度", "伸び", "脆化温度", "監視外れ"]]
    for _, row in selected.iterrows():
        pvc = str(row["primary_pvc"]).split(" (")[0].replace(" / ", "/")
        data.append(
            [
                row["candidate_label"],
                pvc,
                f'{int(row["formulation_material_count"])}',
                f'{row["predicted__S03010_030010"]:.1f}',
                f'{row["predicted__S06060_060030"]:.1f}',
                f'{row["predicted__S13010_130010"]:.1f}',
                f'{int(row["predicted_monitors_violation_count"])}件',
            ]
        )
    return styled_table(data, [15 * mm, 51 * mm, 17 * mm, 18 * mm, 20 * mm, 25 * mm, 22 * mm])


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    slate = pd.read_csv(SLATE)
    errors = json.loads(ERRORS.read_text(encoding="utf-8"))["errors"]

    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title="プラス・テク共同検討 進捗・実施結果報告",
        author="プロジェクトチーム",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="content")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=footer)])

    story = []
    story.append(paragraph("プラス・テク共同検討", STYLES["eyebrow"]))
    story.append(paragraph("進捗・実施結果報告", STYLES["title"]))
    story.append(paragraph("柔軟・低温用途の逆解析 | 2026年10月20日会議", STYLES["subtitle"]))
    story.append(Spacer(1, 8 * mm))
    story.append(
        card(
            "結論",
            "前回から、物性の役割分離、予測誤差、処方制約、3つの選定方針、会議後のレビュー表まで実装・再解析した。分析結果を出すだけでなく、企業側の条件を反映して再解析し、試作判断へつなげる運用まで整備した。",
            PALE_GREEN,
        )
    )
    story.append(Spacer(1, 6 * mm))
    summary = [
        ["仮想候補", "処方制約通過", "物性＋処方通過", "方針別候補", "重複除外後"],
        ["345件", "63件", "16件", "3方針 × 5件", "9候補"],
    ]
    table = styled_table(summary, [34.8 * mm] * 5)
    table.setStyle(TableStyle([("BACKGROUND", (0, 1), (-1, 1), PALE_BLUE), ("ALIGN", (0, 1), (-1, 1), "CENTER")]))
    story.append(table)
    story.append(Spacer(1, 7 * mm))
    story.append(paragraph("前回からの進捗", STYLES["h2"]))
    for text in (
        "脆化温度を最適化、硬度を管理目標、引張伸びを制約、残る3物性を監視へ分離。",
        "各候補へp90予測下限・上限と物性別の保守判定を追加。",
        "材料数・総配合量・主材料特定・禁止原料を設定ファイルで変更可能にした。",
        "性能重視・単純処方重視・監視値重視の候補を各5件選定。",
        "会議後の採否と除外理由を残す候補レビュー表を作成。",
        "自動テスト12件がすべて通過。",
    ):
        story.append(bullet(text))
    story.append(Spacer(1, 5 * mm))
    story.append(
        card(
            "重要な注意",
            "新しい実測データは追加されていない。今回の進捗はモデル精度の向上ではなく、候補の選び方と意思決定プロセスの改善である。",
            PALE_YELLOW,
        )
    )

    story.append(PageBreak())
    story.append(paragraph("1. 前回から実施した作業と成果", STYLES["h1"]))
    story.append(paragraph("前回課題をそのままにせず、分析設計・実装・再解析・会議運用の4段階を進めた。以下が今回の作業と、確認できた結果である。", STYLES["body"]))
    story.append(Spacer(1, 4 * mm))
    progress_data = [
        ["領域", "前回の状態", "今回実施したこと", "結果"],
        ["物性設計", "4物性を同じスコア・合否に使用", "最適化・管理・制約・監視へ分離", "目的・必要条件通過85件"],
        ["最適化", "-41℃以下を同点扱い", "脆化温度を低い方向へ最小化", "改善方向を順位へ反映"],
        ["処方制約", "材料数を個別確認", "JSONで材料数・総量・禁止原料等を設定", "処方通過63件／物性含む16件"],
        ["予測誤差", "平均誤差を説明のみ", "候補別p90上下限・保守判定を追加", "全6物性保守通過0件を明示"],
        ["候補選定", "1種類の上位候補", "性能・単純処方・監視値の3方針化", "各5件／重複除外後9候補"],
        ["会議運用", "統一した採否記録なし", "候補レビュー表を作成", "採否・除外理由等を記録可能"],
        ["品質確認", "自動テスト8件", "制約・誤差判定テストを追加", "12件すべて通過"],
    ]
    story.append(styled_table(progress_data, [24 * mm, 50 * mm, 62 * mm, 38 * mm], font_size=7.0))
    story.append(Spacer(1, 6 * mm))
    story.append(paragraph("作成した仕組みと成果物", STYLES["h2"]))
    artifact_data = [
        ["区分", "内容", "実務上の価値"],
        ["分析ロジック", "物性役割分離・最小化・多様性を考慮した選定", "候補選定理由を説明できる"],
        ["制約エンジン", "材料数・総配合量・主材料・禁止原料", "企業側条件を設定変更だけで反映"],
        ["不確実性", "物性別p90予測幅・保守判定", "予測の過信を防ぎ実測要否を明示"],
        ["会議運用", "3方針候補と9候補のレビュー表", "会議の判断を次の解析へ戻せる"],
        ["再現性", "設定JSON・解析スクリプト・12テスト", "同じ条件で再実行・検証できる"],
    ]
    story.append(styled_table(artifact_data, [31 * mm, 78 * mm, 65 * mm], font_size=7.2))
    story.append(Spacer(1, 6 * mm))
    story.append(
        card(
            "今回の到達点",
            "分析結果を提示するだけでなく、企業側の条件を受け取り、候補を再計算し、採否理由と試作結果を次の改善へ戻す一連の流れを構築した。",
            PALE_GREEN,
        )
    )

    story.append(PageBreak())
    story.append(paragraph("2. 物性の役割と予測誤差", STYLES["h1"]))
    role_data = [
        ["役割", "物性", "現在の条件", "p90絶対誤差"],
        ["最適化", "脆化温度", "-41℃以下を参考に低い方向", f'{errors["S13010_130010"]:.2f}℃'],
        ["管理目標", "硬度 HDA", "70 ± 4", f'{errors["S03010_030010"]:.1f}'],
        ["制約", "引張伸び", "370以上", f'{errors["S06060_060030"]:.1f}'],
        ["監視", "比重", "1.34 ± 0.06", f'{errors["S01010_010010"]:.4f}'],
        ["監視", "100%モジュラス", "7.5 ± 1.5", f'{errors["S06010_060030"]:.2f}'],
        ["監視", "引張強さ", "20以上", f'{errors["S06020_060030"]:.2f}'],
    ]
    story.append(styled_table(role_data, [24 * mm, 47 * mm, 65 * mm, 38 * mm]))
    story.append(Spacer(1, 6 * mm))
    story.append(
        card(
            "保守判定の結果: 全6物性同時通過は0件",
            "硬度の目標許容幅は±4だが、p90絶対誤差は7.8である。このため、点予測が目標内でも誤差幅全体を目標内に収めることはできない。0件は候補が無価値という意味ではなく、試作・実測なしに合格保証できないことを表す。",
            PALE_YELLOW,
        )
    )
    story.append(Spacer(1, 7 * mm))
    story.append(paragraph("設定ファイルで変更できる処方制約", STYLES["h2"]))
    constraint_data = [
        ["設定", "現在値", "会議での確認事項"],
        ["最大材料数", "10", "材料の数え方と上限"],
        ["総配合量 下限／上限", "未設定", "基準量・許容範囲"],
        ["主PVC・主可塑剤の特定", "必須", "名称不明候補の扱い"],
        ["使用禁止原料", "未設定", "禁止・併用不可・法規"],
    ]
    story.append(styled_table(constraint_data, [50 * mm, 35 * mm, 89 * mm]))
    story.append(Spacer(1, 5 * mm))
    story.append(paragraph("制約に違反した候補は削除せず、候補ごとに違反理由を残す。企業側のルール確定後、JSONを変更して同じ解析を再実行できる。", STYLES["body"]))

    story.append(PageBreak())
    story.append(paragraph("3. 3方針の試作候補", STYLES["h1"]))
    story.append(paragraph("物性の目的・必要条件と現在の処方制約を通過した16件から選定した。数値はすべて予測値である。監視外れは、比重・100%モジュラス・引張強さの参考条件から外れる項目数を示す。", STYLES["body"]))
    story.append(Spacer(1, 4 * mm))
    for profile, note in (
        ("性能重視", "脆化温度の低さを最優先"),
        ("単純処方重視", "材料数の少なさを最優先"),
        ("監視値重視", "監視物性の外れ数・距離を最優先"),
    ):
        story.append(KeepTogether([paragraph(f"{profile} - {note}", STYLES["h3"]), profile_table(slate, profile), Spacer(1, 4 * mm)]))
    story.append(Spacer(1, 2 * mm))
    story.append(
        card(
            "候補の読み方",
            "同じ候補が複数方針に選ばれる。15枠の重複を除くと9候補であり、候補レビュー表の候補番号で照合する。順位差の小数点以下を過信せず、異なるPVC系列と判断軸を含む試作構成を優先する。",
            PALE_BLUE,
        )
    )

    story.append(PageBreak())
    story.append(paragraph("4. 会議で決めてほしいこと", STYLES["h1"]))
    decisions = [
        ["優先", "確認事項", "決まるとできること"],
        ["1", "脆化温度を主目的としてよいか", "最適化軸を確定"],
        ["2", "比重・100%M・引張強さは必須か監視か", "候補の合否条件を確定"],
        ["3", "材料数の定義と上限", "単純処方の探索範囲を確定"],
        ["4", "禁止原料・併用不可・総配合量・加工制約", "解析候補を試作可能候補へ変換"],
        ["5", "9候補の採用・保留・除外と理由", "初回試作候補を確定"],
        ["6", "基準品、試作数、測定項目、担当、返却日", "実験計画と再学習工程を確定"],
    ]
    story.append(styled_table(decisions, [14 * mm, 82 * mm, 78 * mm]))
    story.append(Spacer(1, 7 * mm))
    story.append(paragraph("推奨する進め方", STYLES["h2"]))
    for text in (
        "役割分離と『予測は保証ではない』ことを最初に確認する。",
        "性能・単純処方・監視値のどれを優先するか決める。",
        "候補レビュー表を使い、原料・加工・コスト・法規・在庫を確認する。",
        "基準品を含めた初回試作候補を確定する。",
        "実測結果の返却形式、担当者、日付を決める。",
    ):
        story.append(bullet(text))
    story.append(Spacer(1, 6 * mm))
    story.append(
        card(
            "会議冒頭の30秒説明",
            "前回から、物性の役割分離、予測誤差、処方制約、3方針の候補選定まで進めました。現在は16件の実務候補があり、3つの判断軸で9候補へ整理しています。本日は必須物性と処方制約を確認し、レビュー表で初回試作を決めてください。予測は試作順位の参考であり、製品合格の保証ではありません。",
            PALE_GREEN,
        )
    )

    story.append(PageBreak())
    story.append(paragraph("5. 共有物と会議後の更新", STYLES["h1"]))
    share_data = [
        ["共有先", "ファイル", "用途"],
        ["会議参加者全員", "meeting_report_2026-10-20_final.pdf", "説明・意思決定の本体"],
        ["司会・記録担当", "candidate_review_sheet.csv", "採否と除外理由の記録"],
    ]
    story.append(styled_table(share_data, [35 * mm, 77 * mm, 62 * mm]))
    story.append(Spacer(1, 7 * mm))
    story.append(paragraph("会議後の更新手順", STYLES["h2"]))
    for text in (
        "レビュー表へ企業側判断、除外理由、加工性、コスト、法規・在庫、試作優先度を記録する。",
        "合意した材料数、総配合量、禁止原料を制約JSONへ反映する。",
        "再解析し、3方針の候補を更新する。",
        "試作結果が得られたら予測値との差を確認し、誤差参照を更新する。",
    ):
        story.append(bullet(text))
    story.append(Spacer(1, 7 * mm))
    story.append(paragraph("共有しない旧版", STYLES["h2"]))
    story.append(paragraph("progress_report_2026-10-20.pdf と meeting_handoff_2026-10-20.pdf は今回の予測誤差・処方制約・3方針選定を反映する前の版である。最終版と混在させず、原則共有しない。", STYLES["body"]))
    story.append(Spacer(1, 7 * mm))
    story.append(
        card(
            "到達点",
            "AIが最適配合を決定したのではなく、用途価値・必要条件・監視項目・予測誤差・処方制約を分けて、試作候補を説明可能な形で絞り、会議の判断を次の再解析へ戻せるようになった。",
            PALE_BLUE,
        )
    )
    story.append(Spacer(1, 12 * mm))
    story.append(paragraph("再現条件", STYLES["h2"]))
    story.append(paragraph("データ: processed_main（SHA-256: 1c18e064db2a066c71e23df803a7a69c29219a2583bded273d6ea260005bd60b） / モデル: LightGBM / 木: 800 / 乱数: 42 / 誤差: 5-fold OOF p90絶対誤差", STYLES["small"]))
    doc.build(story)


register_fonts()
samples = getSampleStyleSheet()
STYLES = {
    "title": ParagraphStyle("title", fontName="JP-Bold", fontSize=28, leading=36, textColor=NAVY, alignment=TA_LEFT, spaceAfter=4 * mm),
    "subtitle": ParagraphStyle("subtitle", fontName="JP", fontSize=11, leading=17, textColor=GREY),
    "eyebrow": ParagraphStyle("eyebrow", fontName="JP-Bold", fontSize=9, leading=12, textColor=BLUE, spaceAfter=3 * mm),
    "h1": ParagraphStyle("h1", fontName="JP-Bold", fontSize=18, leading=25, textColor=NAVY, spaceAfter=6 * mm),
    "h2": ParagraphStyle("h2", fontName="JP-Bold", fontSize=13, leading=19, textColor=NAVY, spaceBefore=2 * mm, spaceAfter=3 * mm),
    "h3": ParagraphStyle("h3", fontName="JP-Bold", fontSize=10, leading=14, textColor=BLUE, spaceBefore=2 * mm, spaceAfter=2 * mm),
    "body": ParagraphStyle("body", fontName="JP", fontSize=9.2, leading=15, textColor=colors.HexColor("#25333D")),
    "bullet": ParagraphStyle("bullet", fontName="JP", fontSize=9.2, leading=14, leftIndent=4 * mm, firstLineIndent=-3 * mm, spaceAfter=1.5 * mm, textColor=colors.HexColor("#25333D")),
    "card_title": ParagraphStyle("card_title", fontName="JP-Bold", fontSize=11, leading=15, textColor=NAVY),
    "table_header": ParagraphStyle("table_header", fontName="JP-Bold", fontSize=7.6, leading=10, textColor=colors.white, alignment=TA_CENTER),
    "table_cell": ParagraphStyle("table_cell", fontName="JP", fontSize=7.4, leading=10, textColor=colors.HexColor("#25333D")),
    "small": ParagraphStyle("small", fontName="JP", fontSize=7.5, leading=11, textColor=GREY),
}


if __name__ == "__main__":
    build()
