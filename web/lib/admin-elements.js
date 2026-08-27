export function bindAdminElements(doc = document) {
  return {
    adminStatus: doc.getElementById("adminStatus"),
    adminIdentity: doc.getElementById("adminIdentity"),
    adminRefreshAll: doc.getElementById("adminRefreshAll"),
    adminMetrics: doc.getElementById("adminMetrics"),
    testLinkCreate: doc.getElementById("testLinkCreate"),
    testLinkResult: doc.getElementById("testLinkResult"),
    testLinkUrl: doc.getElementById("testLinkUrl"),
    testLinkCopy: doc.getElementById("testLinkCopy"),
    testLinkMeta: doc.getElementById("testLinkMeta"),
    requestStatusFilter: doc.getElementById("requestStatusFilter"),
    adminRequestBoard: doc.getElementById("adminRequestBoard"),
    userSearch: doc.getElementById("userSearch"),
    adminUserTable: doc.getElementById("adminUserTable"),
    benchStatusFilter: doc.getElementById("benchStatusFilter"),
    benchRefresh: doc.getElementById("benchRefresh"),
    adminBenchBoard: doc.getElementById("adminBenchBoard"),
    structuresRefresh: doc.getElementById("structuresRefresh"),
    adminStructuresBoard: doc.getElementById("adminStructuresBoard"),
  };
}
