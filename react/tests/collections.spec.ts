import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

// Generates a unique name to avoid test conflicts.
const getUniqueName = (baseName: string) => `${baseName} ${Date.now()}`;

// Reads test link from 'test-data/test-link.txt'.
const getTestLink = () => {
  const linkFilePath = 'test-data/test-link.txt';
  try {
    const resolvedPath = path.resolve(linkFilePath); // Absolute path for error messages.
    if (!fs.existsSync(resolvedPath)) {
      throw new Error(`File not found at ${resolvedPath}. CWD is ${process.cwd()}`);
    }
    return fs.readFileSync(resolvedPath, 'utf-8').trim();
  } catch (error) {
    console.error(`Failed to read test link from ${path.resolve(linkFilePath)}:`, error);
    // Re-throw for clarity in test output.
    throw new Error(`Could not read test link from ${path.resolve(linkFilePath)}. Ensure file exists and is readable. Error: ${(error as Error).message}`);
  }
};


test.describe('Collection Management and Operations', () => {
  // Assumes user is logged in for these tests.
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:8080/');
    await page.waitForLoadState('networkidle');
    await page.getByRole('link', { name: 'Collections' }).click();
    await expect(page.getByRole('heading', { name: 'My Collections' })).toBeVisible();
  });

  test('should allow creating a new collection', async ({ page }) => {
    const newCollectionName = getUniqueName('My Test Collection');
    await page.getByRole('button', { name: 'New Collection' }).click();
    await expect(page.getByRole('heading', { name: 'Create New Collection' })).toBeVisible();
    await page.getByRole('textbox', { name: 'Collection Name' }).fill(newCollectionName);
    await page.getByRole('button', { name: 'Create' }).click();
    await expect(page.getByRole('heading', { name: newCollectionName, level: 2 })).toBeVisible({ timeout: 10000 });
  });

  test.describe('Operations within an Existing Collection', () => {
    let collectionNameForOps: string;
    // collectionId might be needed later.

    test.beforeEach(async ({ page }) => {
      collectionNameForOps = getUniqueName('Ops Collection');
      await page.getByRole('button', { name: 'New Collection' }).click();
      await expect(page.getByRole('heading', { name: 'Create New Collection' })).toBeVisible();
      await page.getByRole('textbox', { name: 'Collection Name' }).fill(collectionNameForOps);
      await page.getByRole('button', { name: 'Create' }).click();
      const collectionCard = page.locator(`div.bg-scheme-shade_3:has(h2:text-is("${collectionNameForOps}"))`);
      await expect(collectionCard.getByRole('heading', { name: collectionNameForOps, level: 2 })).toBeVisible({ timeout: 10000 });

      await collectionCard.getByRole('heading', { name: collectionNameForOps, level: 2 }).click();
      await expect(page.getByRole('heading', { name: collectionNameForOps, level: 1 })).toBeVisible();
      await expect(page.getByText('Add Content')).toBeVisible();
    });

    test('should allow uploading a PDF, entering its title, and submitting', async ({ page }) => {
      const pdfTitle = getUniqueName('Test PDF Lecture');
      // Assumes tests run from 'react/' directory.
      const pdfFilePath = 'test-data/cpu-intro-2.pdf';

      let pdfIngestionResponseStatus = 0;
      let pdfIngestionResponseBody: any = null;
      let apiRequestMade = false;

      page.on('response', async response => {
        if (response.url().includes('/api/ingest_pdf/') && response.request().method() === 'POST') {
          apiRequestMade = true;
          console.log('--- PDF Ingestion API Response ---');
          console.log('URL:', response.url());
          pdfIngestionResponseStatus = response.status();
          console.log('Status:', pdfIngestionResponseStatus);
          try {
            pdfIngestionResponseBody = await response.json();
            console.log('Body:', JSON.stringify(pdfIngestionResponseBody, null, 2));
          } catch (e) {
            const textBody = await response.text();
            pdfIngestionResponseBody = textBody;
            console.log('Body (not JSON or empty):', textBody);
          }
          console.log('------------------------------------');
        }
      });

      const fileChooserPromise = page.waitForEvent('filechooser');
      // Select PDF type (default, but explicit).
      await page.locator('button[title="PDF"]').click();
      await page.getByText('Select PDF File', { exact: true }).click();
      const fileChooser = await fileChooserPromise;

      // Check if file exists before upload for debugging.
      const resolvedPdfPath = path.resolve(pdfFilePath);
      if (!fs.existsSync(resolvedPdfPath)) {
          throw new Error (`PDF file for upload not found at ${resolvedPdfPath}. CWD: ${process.cwd()}`);
      }
      await fileChooser.setFiles(resolvedPdfPath);


      const titleTextbox = page.getByPlaceholder('Enter PDF title');
      await expect(titleTextbox).toBeVisible({ timeout: 10000 });
      await titleTextbox.fill(pdfTitle);
      await page.getByRole('button', { name: 'Submit All' }).click();

      const successMessageLocator = page.locator('p.text-green:has-text("Submission successful!")');
      const errorMessageLocator = page.locator('p.text-red:text-matches("Error:")');

      try {
        await Promise.race([
            successMessageLocator.waitFor({ state: 'visible', timeout: 25000 }),
            errorMessageLocator.waitFor({ state: 'visible', timeout: 25000 })
        ]);
      } catch (e) {
        console.error('Neither success nor error message for PDF submission appeared in the UI.');
        if (apiRequestMade) {
          console.error(`PDF Ingestion API was called. Status: ${pdfIngestionResponseStatus}, Body: ${JSON.stringify(pdfIngestionResponseBody)}`);
        } else {
          console.error('PDF Ingestion API call was NOT detected.');
        }
        throw new Error(`Timeout waiting for PDF submission UI feedback. API called: ${apiRequestMade}, Status: ${pdfIngestionResponseStatus}`);
      }

      if (await errorMessageLocator.isVisible()) {
          const errorText = await errorMessageLocator.textContent();
          console.error(`PDF Ingestion failed in UI: ${errorText}`);
          throw new Error(`PDF Ingestion failed in UI: ${errorText}. API Status: ${pdfIngestionResponseStatus}, Body: ${JSON.stringify(pdfIngestionResponseBody)}`);
      }
      await expect(successMessageLocator).toBeVisible();

      await page.reload();
      await expect(page.getByRole('heading', { name: collectionNameForOps, level: 1 })).toBeVisible({ timeout: 10000 });
      await expect(page.getByText('Add Content')).toBeVisible();
      await expect(async () => {
        await expect(page.getByRole('cell', { name: pdfTitle, exact: true })).toBeVisible();
      }).toPass({ timeout: 30000 });
    });

    test('should allow ingesting a webpage with crawl depth 0', async ({ page }) => {
        // Set a longer timeout for this specific test due to crawling
        test.setTimeout(2 * 60 * 1000 + 30 * 1000); // 2 minutes 30 seconds (120s for crawl + 30s buffer)
  
        const testURL = getTestLink();
        // This title needs to EXACTLY match what your backend extracts and saves.
        // Manually ingest the URL once and copy the exact title from your application.
        const expectedPageTitleInList = 'Doom: The Dark Ages review – id Software’s return to medieval hell is savage fun';
  
        // Ensure the 'Ops Collection' (collectionNameForOps) is visible and ready
        await expect(page.getByRole('heading', { name: collectionNameForOps, level: 1 })).toBeVisible();
        await expect(page.getByText('Add Content')).toBeVisible();
  
        // 1. Select Webpage ingestion type
        await page.locator('button[title="Webpage"]').click();
        await expect(page.getByRole('textbox', { name: 'Enter Webpage URL' })).toBeVisible(); // Confirm switch
  
        // 2. Fill in URL and crawl depth
        await page.getByRole('textbox', { name: 'Enter Webpage URL' }).fill(testURL);
        await page.getByRole('spinbutton', { name: 'Crawl Depth:' }).fill('0');
  
        // 3. Intercept API response for debugging (optional but good)
        let webpageApiCalled = false;
        let webpageApiResponseStatus: number | undefined;
        page.on('response', async response => {
          if (response.url().includes('/api/ingest_webpage/') && response.request().method() === 'POST') {
              webpageApiCalled = true;
              webpageApiResponseStatus = response.status();
              console.log('--- Webpage Ingestion API Response ---');
              console.log(`URL: ${response.url()}, Status: ${webpageApiResponseStatus}`);
              try {
                  console.log('Body:', JSON.stringify(await response.json(), null, 2));
              } catch (e) {
                  console.log('Body (not JSON or empty):', await response.text());
              }
              console.log('------------------------------------');
          }
        });
  
        // 4. Submit
        await page.getByRole('button', { name: 'Submit All' }).click();
  
        // 5. Verify client-side "initiated" message
        await expect(page.locator('p.text-blue:has-text("Webpage crawl initiated...")')).toBeVisible({ timeout: 15000 });
        // Verify API was called and returned 202 (Accepted)
        await expect.poll(() => webpageApiCalled, { message: "Webpage API call was not detected", timeout: 5000 }).toBe(true);
        expect(webpageApiResponseStatus).toBe(202);
  
  
        // 6. Open and monitor Ingestion Dashboard
        console.log('Opening Ingestion Dashboard to monitor crawl...');
        const launcher = page.getByTestId('ingestion-monitor-launcher');
        await expect(launcher).toBeVisible();
        await launcher.click();
  
        const modalVisibilityWrapper = page.getByTestId('modal-visibility-wrapper');
        await expect(modalVisibilityWrapper).not.toHaveClass(/hidden/, { timeout: 10000 });
  
        await expect(page.getByRole('heading', { name: 'Ingestion Dashboard' })).toBeVisible({ timeout: 5000 });
  
        // Locate the task row in the dashboard.
        // IngestionDashboard.tsx shows: `Crawl: ${task.initialUrl || task.taskId}`
        // and status: <span class="text-xs font-semibold ...">success</span>
        const crawlTaskRowLocator = page.locator('div.p-3', { hasText: `Crawl: ${testURL}` });
        // Ensure the task row itself eventually appears if the dashboard loads asynchronously
        await expect(crawlTaskRowLocator).toBeVisible({ timeout: 15000 });
  
  
        console.log(`Waiting for crawl task for "${testURL}" to show "success" in dashboard...`);
        // Wait for the "success" status within that task's row.
        await expect(crawlTaskRowLocator.locator('span.text-xs.font-semibold:has-text("success")'))
            .toBeVisible({ timeout: 120000 }); // 2 minutes for crawl processing
  
        console.log('Crawl task successful in dashboard. Closing dashboard.');
        // Close the dashboard. The modal has an X button.
        // <button onClick={onClose} ...><X /></button>
        // The modal itself has role="dialog"
        const modalDialog = page.getByRole('dialog', { name: 'Ingestion Dashboard' }); // More specific
        await modalDialog.getByRole('button', { name: 'Close ingestion dashboard' }).click(); // Using aria-label
  
        await expect(page.getByRole('heading', { name: 'Ingestion Dashboard' })).not.toBeVisible({ timeout: 5000 });
  
        // 7. Reload CollectionView and verify ingested document
        console.log('Reloading collection view...');
        await page.reload();
        await expect(page.getByRole('heading', { name: collectionNameForOps, level: 1 })).toBeVisible({ timeout: 10000 });
        await expect(page.getByText('Add Content')).toBeVisible(); // Confirm page loaded
  
        console.log(`Verifying ingested document "${expectedPageTitleInList}" appears in the list...`);
        await expect(async () => {
          // The title in the list might be slightly different or truncated
          await expect(page.getByRole('cell', { name: expectedPageTitleInList, exact: false })).toBeVisible();
        }).toPass({ timeout: 30000 });
  
        console.log('Webpage ingestion test completed successfully.');
      });


    test('should open an ingested document from collection view', async ({ page }) => {
      const docTitleToOpen = getUniqueName('DocToOpenForViewing');
      const pdfFilePath = 'test-data/cpu-intro-2.pdf'; // Relative to CWD (e.g., react/).

      // Ingest a document for this test.
      await page.locator('button[title="PDF"]').click(); // Ensure PDF type.
      const fileChooserPromise = page.waitForEvent('filechooser');
      await page.getByText('Select PDF File', { exact: true }).click();
      const fileChooser = await fileChooserPromise;
      const resolvedPdfPath = path.resolve(pdfFilePath); // Resolved path for reliable file setting.
       if (!fs.existsSync(resolvedPdfPath)) {
          throw new Error (`PDF file for 'open document' test not found at ${resolvedPdfPath}. CWD: ${process.cwd()}`);
      }
      await fileChooser.setFiles(resolvedPdfPath);
      await page.getByPlaceholder('Enter PDF title').fill(docTitleToOpen);
      await page.getByRole('button', { name: 'Submit All' }).click();
      await expect(page.locator('p.text-green:has-text("Submission successful!")')).toBeVisible({timeout: 20000});

      await page.reload();
      await expect(page.getByRole('heading', { name: collectionNameForOps, level: 1 })).toBeVisible({ timeout: 10000 });
      await expect(page.getByText('Add Content')).toBeVisible();
      const documentCell = page.getByRole('cell', { name: docTitleToOpen, exact: true });
      await expect(documentCell).toBeVisible({timeout: 20000});

      // Click document name cell to open.
      await documentCell.click();

      // Verify navigation to document detail page.
      // URL pattern: /document/some-uuid/
      await expect(page).toHaveURL(/\/document\/[0-9a-fA-F-]+\/?$/, { timeout: 10000 });

      // Check for unique element on document detail page (e.g., H1 title).
      // Replace with actual assertion for document detail page content.
      await expect(page.getByRole('heading', { name: docTitleToOpen, level: 1 })).toBeVisible({ timeout: 10000 });
      // Example: If page shows "Document Content:".
      // await expect(page.getByText("Document Content:", { exact: false })).toBeVisible();
    });

  });

  test('should navigate into an existing collection detail page', async ({ page }) => {
    const collectionToNavigate = getUniqueName('NavTest Collection');
    await page.getByRole('button', { name: 'New Collection' }).click();
    await expect(page.getByRole('heading', { name: 'Create New Collection' })).toBeVisible();
    await page.getByRole('textbox', { name: 'Collection Name' }).fill(collectionToNavigate);
    await page.getByRole('button', { name: 'Create' }).click();
    await expect(page.getByRole('heading', { name: collectionToNavigate, level: 2 })).toBeVisible({ timeout: 10000 });

    await page.getByRole('heading', { name: collectionToNavigate, level: 2 }).click();

    await expect(page.getByRole('heading', { name: collectionToNavigate, level: 1 })).toBeVisible();
    await expect(page.getByText('Add Content')).toBeVisible();
    await expect(page.getByText('Select PDF File', { exact: true })).toBeVisible();
  });

  test('should allow moving a collection into another collection', async ({ page }) => {
    const sourceCollectionName = getUniqueName('Source Collection Move Test');
    const destinationCollectionName = getUniqueName('Destination Collection Move Test');

    await page.getByRole('button', { name: 'New Collection' }).click();
    await page.getByRole('textbox', { name: 'Collection Name' }).fill(sourceCollectionName);
    await page.getByRole('button', { name: 'Create' }).click();
    await expect(page.getByRole('heading', { name: sourceCollectionName, level: 2 })).toBeVisible({ timeout: 10000 });

    await page.getByRole('button', { name: 'New Collection' }).click();
    await page.getByRole('textbox', { name: 'Collection Name' }).fill(destinationCollectionName);
    await page.getByRole('button', { name: 'Create' }).click();
    await expect(page.getByRole('heading', { name: destinationCollectionName, level: 2 })).toBeVisible({ timeout: 10000 });

    const sourceCollectionCard = page.locator(
      `div.bg-scheme-shade_3:has(h2:text-is("${sourceCollectionName}"))`
    );
    await expect(sourceCollectionCard).toBeVisible({ timeout: 10000 });

    const settingsButton = sourceCollectionCard.locator('button:has(svg[width="20"])');
    await expect(settingsButton).toBeVisible({ timeout: 5000 });
    await settingsButton.click();
    
    const moveCollectionButtonInMenu = page.getByRole('button', { name: 'Move Collection' });
    await expect(moveCollectionButtonInMenu).toBeVisible({timeout: 5000});
    await moveCollectionButtonInMenu.click();
    
    await expect(page.getByRole('heading', { name: 'Move Collection', level: 3 })).toBeVisible({timeout: 5000});
    
    const destinationListItem = page.locator(`ul li:has-text("${destinationCollectionName}")`);
    await destinationListItem.scrollIntoViewIfNeeded();
    await expect(destinationListItem).toBeVisible();
    await destinationListItem.click();
    
    await page.getByRole('button', { name: 'Select This Location' }).click();

    await expect(page.getByRole('heading', { name: 'Move Collection', level: 3 })).not.toBeVisible({ timeout: 10000 });

    await page.reload();
    await expect(page.getByRole('heading', { name: 'My Collections' })).toBeVisible();

    await expect(page.getByRole('heading', { name: sourceCollectionName, level: 2 })).not.toBeVisible({ timeout: 10000 });

    await page.getByRole('heading', { name: destinationCollectionName, level: 2 }).click();

    await expect(page.getByRole('heading', { name: destinationCollectionName, level: 1 })).toBeVisible({ timeout: 10000 });
    await expect(page.getByText('Add Content')).toBeVisible();

    await expect(async () => {
      await expect(page.getByRole('cell', { name: sourceCollectionName, exact: true })).toBeVisible();
    }).toPass({ timeout: 25000 });
  });

  test('should allow deleting an existing collection', async ({ page }) => {
    const collectionNameToDelete = getUniqueName('Collection To Delete');

    await page.getByRole('button', { name: 'New Collection' }).click();
    await expect(page.getByRole('heading', { name: 'Create New Collection' })).toBeVisible();
    await page.getByRole('textbox', { name: 'Collection Name' }).fill(collectionNameToDelete);
    await page.getByRole('button', { name: 'Create' }).click();
    await expect(page.getByRole('heading', { name: collectionNameToDelete, level: 2 })).toBeVisible({ timeout: 10000 });

    const collectionCardToDelete = page.locator(
      `div.bg-scheme-shade_3:has(h2:text-is("${collectionNameToDelete}"))`
    );
    await expect(collectionCardToDelete).toBeVisible();

    const settingsButton = collectionCardToDelete.locator('button:has(svg[width="20"])');
    await expect(settingsButton).toBeVisible();
    await settingsButton.click();

    const deleteButtonInMenu = page.locator('div[style*="position: absolute"] button:has-text("Delete")');
    await expect(deleteButtonInMenu).toBeVisible({ timeout: 5000 });

    let dialogAccepted = false;
    page.once('dialog', async dialog => {
      console.log(`Dialog message for delete: ${dialog.message()}`);
      expect(dialog.message()).toContain(`Are you sure you want to delete "${collectionNameToDelete}"?`);
      await dialog.accept();
      dialogAccepted = true;
    });

    await deleteButtonInMenu.click();

    await expect.poll(() => dialogAccepted, { timeout: 5000 }).toBe(true);
    
    await page.waitForTimeout(500); // Brief pause after dialog.
    await page.reload();
    await expect(page.getByRole('heading', { name: 'My Collections' })).toBeVisible();

    await expect(page.getByRole('heading', { name: collectionNameToDelete, level: 2 })).not.toBeVisible({ timeout: 10000 });
  });
});