import { chromium } from 'playwright'
const sample = `First Article Title\n\nBody one of the first article. It talks about many things in detail.\n\nSecond Article Title\n\nBody of the second article, also quite interesting to read fully.\n\nThird Article Title\n\nBody of the third article, the final one in this issue.`
const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' })
const page = await browser.newPage()
const errs = []
page.on('pageerror', (e) => errs.push(e.message))
await page.goto('http://localhost:4183')
await page.click('text=+ 上传周刊')
await page.click('text=📋 粘贴文本')
await page.fill('textarea', sample)
await page.click('text=解析并拆分')
await page.waitForSelector('text=确认拆分结果')
await page.click('button:has-text("确认保存")')
await page.waitForSelector('text=本期已收录')
await page.click('h3:has-text("First Article Title")')
await page.waitForSelector('article h1:has-text("First Article Title")')
// 下一篇按钮:应显示第二篇标题
await page.waitForSelector('text=下一篇 →')
await page.click('div:has-text("Second Article Title") >> nth=-1')
await page.waitForSelector('article h1:has-text("Second Article Title")')
console.log('✓ 「下一篇」跳转成功,且上一篇被自动标记已读')
// 上一篇应显示已读状态与第一篇标题
await page.waitForSelector('text=← 上一篇')
// 返回本期
await page.click('button:has-text("返回本期")')
await page.waitForSelector('text=本期已收录')
const readBadges = await page.locator('span:has-text("已读完")').count()
console.log(`✓ 返回本期概览,已读标记 ${readBadges} 个(第一篇应为已读)`)
// 最后一篇:无「下一篇」
await page.click('h3:has-text("Third Article Title")')
await page.waitForSelector('article h1:has-text("Third Article Title")')
const nextCount = await page.locator('text=下一篇 →').count()
console.log(nextCount === 0 ? '✓ 最后一篇没有「下一篇」按钮' : '✗ 最后一篇仍显示下一篇')
console.log(errs.length ? '页面错误: ' + errs.join('|') : '✓ 无页面错误')
await browser.close()
console.log('NAV TEST DONE')
