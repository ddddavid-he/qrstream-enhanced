export function formatSearch(data: any, q: string): string {
  const lines: string[] = [];
  const count = data.organic?.length ?? 0;
  lines.push(`🔍 Google Search: "${q}" (${count} results, 1 credit)`);

  if (data.answerBox) {
    const ab = data.answerBox;
    lines.push('', `💡 Answer: ${ab.answer || ab.snippet || ab.title || ''}`);
    if (ab.link) lines.push(`   ${ab.link}`);
  }

  if (data.knowledgeGraph) {
    const kg = data.knowledgeGraph;
    lines.push('', `📦 Knowledge Graph: ${kg.title}${kg.type ? ' — ' + kg.type : ''}`);
    if (kg.website) lines.push(`   ${kg.website}`);
    if (kg.description) lines.push(`   "${kg.description}"`);
  }

  if (data.organic?.length) {
    lines.push('', '📋 Results:');
    for (const r of data.organic) {
      lines.push(`  ${r.position}. ${r.title}`);
      lines.push(`     ${r.link}`);
      if (r.snippet) lines.push(`     ${r.snippet}`);
      if (r.date) lines.push(`     📅 ${r.date}`);
      lines.push('');
    }
  }

  if (data.peopleAlsoAsk?.length) {
    lines.push('❓ People Also Ask:');
    for (const p of data.peopleAlsoAsk) lines.push(`  • ${p.question}`);
  }

  if (data.relatedSearches?.length) {
    lines.push('', `🔗 Related: ${data.relatedSearches.map((r: any) => `"${r.query}"`).join(', ')}`);
  }

  if (data.credits !== undefined) lines.push('', `💰 Balance: ${data.credits.toLocaleString()} credits`);
  return lines.join('\n');
}

export function formatNews(data: any, q: string): string {
  const lines: string[] = [];
  const count = data.news?.length ?? 0;
  lines.push(`📰 Google News: "${q}" (${count} results, 1 credit)`);
  if (data.news?.length) {
    lines.push('');
    for (const [i, n] of data.news.entries()) {
      lines.push(`  ${i + 1}. ${n.title}`);
      lines.push(`     ${n.link}`);
      if (n.source) lines.push(`     📰 ${n.source}${n.date ? ' · ' + n.date : ''}`);
      if (n.snippet) lines.push(`     ${n.snippet}`);
      lines.push('');
    }
  }
  if (data.credits !== undefined) lines.push(`💰 Balance: ${data.credits.toLocaleString()} credits`);
  return lines.join('\n');
}

export function formatImages(data: any, q: string): string {
  const lines: string[] = [];
  const count = data.images?.length ?? 0;
  lines.push(`🖼️ Google Images: "${q}" (${count} results, 1 credit)`);
  if (data.images?.length) {
    lines.push('');
    for (const [i, img] of data.images.entries()) {
      lines.push(`  ${i + 1}. ${img.title}`);
      lines.push(`     🔗 ${img.link}`);
      lines.push(`     🖼️ ${img.imageUrl}`);
      if (img.source) lines.push(`     📰 ${img.source}`);
      lines.push('');
    }
  }
  if (data.credits !== undefined) lines.push(`💰 Balance: ${data.credits.toLocaleString()} credits`);
  return lines.join('\n');
}

export function formatVideos(data: any, q: string): string {
  const lines: string[] = [];
  const count = data.videos?.length ?? 0;
  lines.push(`🎬 Google Videos: "${q}" (${count} results, 1 credit)`);
  if (data.videos?.length) {
    lines.push('');
    for (const [i, v] of data.videos.entries()) {
      lines.push(`  ${i + 1}. ${v.title}`);
      lines.push(`     ${v.link}`);
      if (v.channel) lines.push(`     📺 ${v.channel}${v.duration ? ' · ' + v.duration : ''}${v.date ? ' · ' + v.date : ''}`);
      if (v.snippet) lines.push(`     ${v.snippet}`);
      lines.push('');
    }
  }
  if (data.credits !== undefined) lines.push(`💰 Balance: ${data.credits.toLocaleString()} credits`);
  return lines.join('\n');
}

export function formatPlaces(data: any, q: string): string {
  const lines: string[] = [];
  const count = data.places?.length ?? 0;
  lines.push(`📍 Google Places: "${q}" (${count} results, 1 credit)`);
  if (data.places?.length) {
    lines.push('');
    for (const [i, p] of data.places.entries()) {
      lines.push(`  ${i + 1}. ${p.title}${p.category ? ' (' + p.category + ')' : ''}`);
      if (p.address) lines.push(`     📍 ${p.address}`);
      if (p.rating) lines.push(`     ⭐ ${p.rating}${p.ratingCount ? ' (' + p.ratingCount + ' reviews)' : ''}`);
      if (p.phoneNumber) lines.push(`     📞 ${p.phoneNumber}`);
      if (p.website) lines.push(`     🔗 ${p.website}`);
      lines.push('');
    }
  }
  if (data.credits !== undefined) lines.push(`💰 Balance: ${data.credits.toLocaleString()} credits`);
  return lines.join('\n');
}

export function formatShopping(data: any, q: string): string {
  const lines: string[] = [];
  const count = data.shopping?.length ?? 0;
  lines.push(`🛒 Google Shopping: "${q}" (${count} results, ⚠️ 2 credits)`);
  if (data.shopping?.length) {
    lines.push('');
    for (const [i, s] of data.shopping.entries()) {
      lines.push(`  ${i + 1}. ${s.title}`);
      if (s.price) lines.push(`     💲 ${s.price}`);
      if (s.source) lines.push(`     🏪 ${s.source}`);
      if (s.rating) lines.push(`     ⭐ ${s.rating}${s.ratingCount ? ' (' + s.ratingCount + ')' : ''}`);
      if (s.link) lines.push(`     🔗 ${s.link}`);
      lines.push('');
    }
  }
  if (data.credits !== undefined) lines.push(`💰 Balance: ${data.credits.toLocaleString()} credits`);
  return lines.join('\n');
}

export function formatScholar(data: any, q: string): string {
  const lines: string[] = [];
  const count = data.organic?.length ?? 0;
  lines.push(`🎓 Google Scholar: "${q}" (${count} results, 1 credit)`);
  if (data.organic?.length) {
    lines.push('');
    for (const [i, r] of data.organic.entries()) {
      lines.push(`  ${i + 1}. ${r.title}`);
      lines.push(`     ${r.link}`);
      if (r.publicationInfo) lines.push(`     📄 ${r.publicationInfo}`);
      if (r.year) lines.push(`     📅 ${r.year}${r.citedBy ? ' · Cited by ' + r.citedBy : ''}`);
      if (r.snippet) lines.push(`     ${r.snippet}`);
      if (r.pdfUrl) lines.push(`     📥 PDF: ${r.pdfUrl}`);
      lines.push('');
    }
  }
  if (data.credits !== undefined) lines.push(`💰 Balance: ${data.credits.toLocaleString()} credits`);
  return lines.join('\n');
}

export function formatPatents(data: any, q: string): string {
  const lines: string[] = [];
  const count = data.organic?.length ?? 0;
  lines.push(`📜 Google Patents: "${q}" (${count} results, 1 credit)`);
  if (data.organic?.length) {
    lines.push('');
    for (const [i, r] of data.organic.entries()) {
      lines.push(`  ${i + 1}. ${r.title}`);
      if (r.publicationNumber) lines.push(`     📋 ${r.publicationNumber}`);
      if (r.inventor) lines.push(`     👤 ${r.inventor}`);
      if (r.assignee) lines.push(`     🏢 ${r.assignee}`);
      if (r.filingDate) lines.push(`     📅 Filed: ${r.filingDate}${r.grantDate ? ' · Granted: ' + r.grantDate : ''}`);
      if (r.snippet) lines.push(`     ${r.snippet}`);
      lines.push('');
    }
  }
  if (data.credits !== undefined) lines.push(`💰 Balance: ${data.credits.toLocaleString()} credits`);
  return lines.join('\n');
}

export function formatSuggest(data: any, q: string): string {
  const lines: string[] = [];
  lines.push(`💭 Suggestions for: "${q}"`);
  if (data.suggestions?.length) {
    for (const s of data.suggestions) lines.push(`  • ${s.value}`);
  }
  if (data.credits !== undefined) lines.push('', `💰 Balance: ${data.credits.toLocaleString()} credits`);
  return lines.join('\n');
}

export function formatCredits(data: any): string {
  return `💰 Account\n  Balance: ${data.balance?.toLocaleString() ?? '?'} credits\n  Rate Limit: ${data.rateLimit ?? '?'} req/sec`;
}
